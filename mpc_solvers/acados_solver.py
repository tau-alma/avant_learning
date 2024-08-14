import numpy as np
import casadi as cs
import scipy
import torch
import config
from acados_template import AcadosOcp, AcadosOcpSolver
from mpc_solvers.mpc_problem import SymbolicMPCProblem
from typing import List


class AcadosSolver:
    def __init__(self, problem: SymbolicMPCProblem, rebuild: bool=False):
        self.problem = problem

        self.N = problem.N
        self.sample_time = problem.h
        self.time_steps = np.tile(self.sample_time, self.N+1)
        self.tf = self.time_steps.sum()

        self.input_delay = problem.input_delay
        self.delay_compensation_f = None

        self.n_controls = 0
        self.n_states = 0
        self.n_params = 0
        self.n_terminal_param_pad = 0
        
        self.ocp = None
        self.acados_solver = None
        self._create_solver(problem)
        self.initialized = False

        self.xstar = np.zeros([self.N+1, self.n_states])
        self.ustar = np.zeros([self.N, self.n_controls])

    def _create_solver(self, problem: SymbolicMPCProblem):
        self.ocp = AcadosOcp()

        self.ocp.model.name = "acados_ocp"
        self.ocp.dims.N = problem.N
        # States
        self.ocp.model.x = problem.ocp_x
        # Controls
        self.ocp.model.u = problem.ocp_u
        # Parameters
        params = problem.ocp_p

        if problem.has_stage_neural_cost:
            params = cs.vertcat(params, problem.stage_cost_params)
            self.n_terminal_param_pad += problem.stage_cost_params.size1()
        self.ocp.model.p = params

        self.ocp.model.f_expl_expr = problem.dynamics_fun(problem.ocp_x, problem.ocp_u, problem.ocp_p)
        x_dot = cs.MX.sym("xdot", problem.ocp_x.size())
        self.ocp.model.f_impl_expr = x_dot - self.ocp.model.f_expl_expr

        # Ugly way to write delay comp, using RK4, could probably use the integrator of the acadosOcp instead?
        k1 = problem.dynamics_fun(problem.ocp_x, problem.ocp_u, problem.ocp_p)
        k2 = problem.dynamics_fun(problem.ocp_x + self.input_delay / 2 * k1, problem.ocp_u, problem.ocp_p) 
        k3 = problem.dynamics_fun(problem.ocp_x + self.input_delay / 2 * k2, problem.ocp_u, problem.ocp_p)
        k4 = problem.dynamics_fun(problem.ocp_x + self.input_delay * k3, problem.ocp_u, problem.ocp_p)
        dynamics = self.input_delay / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        self.delay_compensation_f = cs.Function("delay_comp", [problem.ocp_x, problem.ocp_u, problem.ocp_p], [problem.ocp_x + dynamics])

        # State constraints:
        self.ocp.constraints.lbx = problem.lbx_vec
        self.ocp.constraints.ubx = problem.ubx_vec
        self.ocp.constraints.lbx_e = problem.lbx_vec
        self.ocp.constraints.ubx_e = problem.ubx_vec
        self.ocp.constraints.idxbx = np.arange(problem.ocp_x.size()[0])
        self.ocp.constraints.idxbx_e = self.ocp.constraints.idxbx
        # State soft constraints:
        if problem.ocp_x_slacks is not None:
            self.ocp.constraints.lsbx = np.zeros(len(problem.ocp_x_slacks))
            self.ocp.constraints.usbx = np.zeros(len(problem.ocp_x_slacks))
            self.ocp.constraints.lsbx_e = self.ocp.constraints.lsbx
            self.ocp.constraints.usbx_e = self.ocp.constraints.usbx
            self.ocp.constraints.idxsbx = np.array([*list(problem.ocp_x_slacks.keys())])
            self.ocp.constraints.idxsbx_e = self.ocp.constraints.idxsbx
            state_slack_weights = np.array([*list(problem.ocp_x_slacks.values())])

        # Control constraints:
        self.ocp.constraints.lbu = problem.lbu_vec
        self.ocp.constraints.ubu = problem.ubu_vec
        self.ocp.constraints.idxbu = np.arange(problem.ocp_u.size()[0])

        # Nonlinear constraints:
        nonlinear_slack_weights = np.array([])
        nonlinear_slack_weights_e = np.array([])
        if problem.g_fun is not None:
            self.ocp.constraints.lh = problem.lbg_vec
            self.ocp.constraints.uh = problem.ubg_vec
            self.ocp.model.con_h_expr = problem.g_fun(problem.ocp_x, problem.ocp_u, problem.ocp_p)
            self.ocp.constraints.idxsh = np.array([*list(problem.ocp_g_slacks.keys())])
            nonlinear_slack_weights = np.array([*list(problem.ocp_g_slacks.values())])
        if problem.terminal_g_fun is not None:
            self.ocp.model.con_h_expr_e = problem.terminal_g_fun(problem.ocp_x, problem.ocp_p)
            self.ocp.constraints.idxsh_e = np.arange(0)
            nonlinear_slack_weights_e = np.array([])

        # Slack penalties:
        self.ocp.cost.zl =   np.r_[state_slack_weights, nonlinear_slack_weights]
        self.ocp.cost.zl_e = np.r_[state_slack_weights, nonlinear_slack_weights_e]
        self.ocp.cost.zu =   np.r_[state_slack_weights, nonlinear_slack_weights]
        self.ocp.cost.zu_e = np.r_[state_slack_weights, nonlinear_slack_weights_e]
        self.ocp.cost.Zl =   np.r_[state_slack_weights, nonlinear_slack_weights]
        self.ocp.cost.Zl_e = np.r_[state_slack_weights, nonlinear_slack_weights_e]
        self.ocp.cost.Zu =   np.r_[state_slack_weights, nonlinear_slack_weights]
        self.ocp.cost.Zu_e = np.r_[state_slack_weights, nonlinear_slack_weights_e]

        # Stage cost:
        if problem.has_stage_neural_cost:
            self.ocp.cost.cost_type = 'EXTERNAL'
            neural_cost_state = problem.ocp_x_to_stage_state_fun(problem.ocp_x, problem.ocp_u, problem.ocp_p)
            self.ocp.model.cost_expr_ext_cost = problem.stage_cost_fun(neural_cost_state, problem.ocp_x, problem.ocp_u, problem.stage_cost_params, problem.ocp_p)
        else:
            self.ocp.cost.cost_type = 'NONLINEAR_LS'
            self.ocp.model.cost_y_expr = problem.stage_cost_fun(problem.ocp_x, problem.ocp_u, problem.ocp_p)
            self.ocp.cost.yref = np.zeros(self.ocp.model.cost_y_expr.size1()) 
            self.ocp.cost.W = scipy.linalg.block_diag(*[1] * self.ocp.model.cost_y_expr.size1())

        # Terminal cost
        if problem.terminal_cost_fun is not None:
            if problem.has_terminal_neural_cost:
                self.ocp.cost.cost_type_e = 'EXTERNAL'
                neural_cost_state = problem.ocp_x_to_terminal_state_fun(problem.ocp_x, problem.ocp_p)
                self.ocp.model.cost_expr_ext_cost_e = problem.terminal_cost_fun(neural_cost_state, problem.ocp_p)
            else:
                self.ocp.cost.cost_type_e = 'NONLINEAR_LS'
                self.ocp.model.cost_y_expr_e = problem.terminal_cost_fun(problem.ocp_x, problem.ocp_p)
                self.ocp.cost.yref_e = np.zeros(self.ocp.model.cost_y_expr_e.size1()) 
                self.ocp.cost.W_e = scipy.linalg.block_diag(*[1] * self.ocp.model.cost_y_expr_e.size1())
        
        # Initialize initial conditions:
        self.n_states = self.ocp.model.x.size()[0]
        self.n_controls = self.ocp.model.u.size()[0]
        self.n_parameters = self.ocp.model.p.size()[0]
        self.ocp.constraints.x0 = np.zeros(self.n_states)
        self.ocp.parameter_values = np.zeros(self.n_parameters)

        # Integrator settings:
        self.ocp.solver_options.time_steps = self.time_steps
        self.ocp.solver_options.tf = self.tf
        self.ocp.solver_options.sim_method_num_stages = 4
        self.ocp.solver_options.sim_method_num_steps = 3
        self.ocp.solver_options.integrator_type = "ERK"

        # Solver settings:
        self.ocp.solver_options.nlp_solver_type = "SQP_RTI"
        self.ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
        self.ocp.solver_options.regularize_method = "PROJECT"
        self.ocp.solver_options.tol = 1e-4
        self.ocp.solver_options.print_level = 0

        self.ocp.solver_options.qp_solver_iter_max = 100
        self.ocp.qp_solver_warm_start = 2
        self.ocp.solver_options.hpipm_mode = 'SPEED'
        if problem.terminal_model_external_shared_lib_dir is not None and problem.terminal_model_external_shared_lib_name is not None:
            self.ocp.solver_options.model_external_shared_lib_dir = problem.terminal_model_external_shared_lib_dir
            self.ocp.solver_options.model_external_shared_lib_name = problem.terminal_model_external_shared_lib_name + ' -l' + problem.terminal_model_external_shared_lib_name
        
        if problem.has_stage_neural_cost or problem.has_terminal_neural_cost:
            self.ocp.solver_options.hessian_approx = "EXACT"
        else:
            self.ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        self.ocp.solver_options.levenberg_marquardt = 1e-4
        self.ocp.solver_options.nlp_solver_step_length = 0.75

        self.acados_solver = AcadosOcpSolver(self.ocp, json_file="acados_ocp.json")

    def _shift_horizon(self):
        self.xstar[:-1] = self.xstar[1:]
        self.ustar[:-1] = self.ustar[1:]
        self.ustar[-1, :] = 0

    def _delay_compensation(self, initial_state):
        new_state = self.delay_compensation_f(initial_state, np.zeros_like(self.ustar[1]), np.empty(0))
        new_state = np.asarray(new_state).flatten()
        return new_state

    def solve(self, x_initial: np.ndarray, params: List[np.ndarray]):
        x_initial = self._delay_compensation(x_initial)
        self.acados_solver.set(0, "lbx", x_initial)
        self.acados_solver.set(0, "ubx", x_initial)

        if not self.initialized:
            self.xstar[:] = x_initial
            self.ustar = np.zeros_like(self.ustar)
        else:
            self.xstar[0] = x_initial

        # Prepare reference target using policy:
        if self.problem.policy is not None:
            reference_targets = []
            state_targets = []
            running_targets = x_initial[-2:]
            running_state = x_initial[:-2]
            for n in range(0, self.N+1):
                padded_params = np.r_[params[n], np.zeros(2)]
                full_state = np.r_[running_state, running_targets]
                policy_input = self.problem.ocp_x_to_policy_state_fun(full_state, padded_params).full()
                policy_input = torch.from_numpy(policy_input).T.to(torch.float32)
                with torch.no_grad():
                    targets = self.problem.policy(policy_input).numpy()[0] * [config.avant_max_dot_beta, config.avant_max_v]

                # Rate limit the change of beta velocity target command by max allowed acceleration:
                new_beta_vel = max(
                    min(
                        targets[0] * config.avant_max_dot_beta,
                        running_targets[0] + self.problem.h * config.avant_max_dot_dot_beta
                    ),
                    running_targets[0] - self.problem.h * config.avant_max_dot_dot_beta
                )
                targets[0] = new_beta_vel

                # Rate limit the change of linear velocity target command by max allowed acceleration:
                new_vel_f = max(
                    min(
                        targets[1] * config.avant_max_v,
                        running_targets[1] + self.problem.h * config.avant_max_a
                    ),
                    running_targets[1] - self.problem.h * config.avant_max_a
                )
                targets[1] = new_vel_f

                running_targets = targets
                running_state = running_state + self.problem.h * self.problem.sim_fun(running_state, targets).full().T[0]
                reference_targets.append(targets)
                state_targets.append(running_state)
            reference_targets = np.asarray(reference_targets)
            state_targets = np.c_[np.asarray(state_targets), reference_targets]
        else:
            reference_targets = np.zeros((self.N+1, 2))

        params = np.c_[params, reference_targets]
        stage_params = np.asarray(params[:-1])
        terminal_params = np.asarray(params[-1])

        # Prepare stage cost linearization params:
        if self.problem.get_stage_cost_params_fun is not None:
            stage_cost_input = self.problem.ocp_x_to_stage_state_fun.map(self.N)(self.xstar[:-1, :].T, self.ustar.T, stage_params.T).full().T
            stage_cost_params = self.problem.get_stage_cost_params_fun(stage_cost_input)
            stage_params = np.c_[stage_params, stage_cost_params]

        # Fill in x0, u0 and p for the solver:
        for i in range(self.N + 1):
            self.acados_solver.set(i, "x", self.xstar[i])
            if i < self.N:
                self.acados_solver.set(i, "u", self.ustar[i])
                self.acados_solver.set(i, "p", np.r_[stage_params[i]])
            else:
                self.acados_solver.set(i, "p", np.r_[terminal_params, np.zeros(self.n_terminal_param_pad)])

        for i in range(25):
            status = self.acados_solver.solve()

        if status in [0, 2]:   # Success or timeout
            self.initialized = True
            for i in range(self.N+1):
                x = self.acados_solver.get(i, "x")
                self.xstar[i] = x
                if i < self.N:
                    u = self.acados_solver.get(i, "u")
                    self.ustar[i] = u
        else:   # Probably infeasibility in underlying QP solver
            self.initialized = False
            print("STATUS", status)
            print(self.acados_solver.get_cost())
            print(self.acados_solver.get_residuals())
            print()

        state_horizon = self.xstar.copy()
        control_horizon = self.ustar.copy()
        self._shift_horizon()

        return state_horizon, control_horizon, status in [0, 2]