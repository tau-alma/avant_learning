import numpy as np
import casadi as cs
import scipy
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

        self.n_controls = 0
        self.n_states = 0
        self.n_params = 0
        self.n_param_pad = 0
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
        self.n_terminal_param_pad += problem.ocp_p.size1()
        if problem.terminal_ocp_p is not None:
            params = cs.vertcat(params, problem.terminal_ocp_p)
            self.n_param_pad += problem.terminal_ocp_p.size1()
        if problem.terminal_cost_params is not None:
            params = cs.vertcat(params, problem.terminal_cost_params)
            self.n_param_pad += problem.terminal_cost_params.size1()
        self.ocp.model.p = params

        self.ocp.model.f_expl_expr = problem.dynamics_fun(problem.ocp_x, problem.ocp_u, problem.ocp_p)
        x_dot = cs.MX.sym("xdot", problem.ocp_x.size())
        self.ocp.model.f_impl_expr = x_dot - self.ocp.model.f_expl_expr

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
        else:
            state_slack_weights = np.array([])

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
        self.ocp.cost.cost_type = 'NONLINEAR_LS'
        self.ocp.model.cost_y_expr = problem.cost_fun(problem.ocp_x, problem.ocp_u, problem.ocp_p)
        self.ocp.cost.yref = np.zeros(self.ocp.model.cost_y_expr.size1()) 
        self.ocp.cost.W = scipy.linalg.block_diag(*[0] * self.ocp.model.cost_y_expr.size1())

        # Terminal cost
        if problem.terminal_cost_fun is not None:
            if problem.has_neural_cost:
                self.ocp.cost.cost_type_e = 'EXTERNAL'
                neural_cost_state = problem.ocp_x_to_terminal_state_fun(problem.ocp_x, problem.terminal_ocp_p)
                if problem.terminal_cost_params is not None:
                    self.ocp.model.cost_expr_ext_cost_e = problem.terminal_cost_fun(neural_cost_state, problem.terminal_cost_params, problem.terminal_ocp_p)
                else:
                    self.ocp.model.cost_expr_ext_cost_e = problem.terminal_cost_fun(neural_cost_state, problem.terminal_ocp_p)
            else:
                self.ocp.cost.cost_type_e = 'NONLINEAR_LS'
                self.ocp.model.cost_y_expr_e = problem.terminal_cost_fun(problem.ocp_x, problem.ocp_p)
                self.ocp.cost.yref_e = np.zeros(self.ocp.model.cost_y_expr_e.size1()) 
                self.ocp.cost.W_e = scipy.linalg.block_diag(*[0] * self.ocp.model.cost_y_expr_e.size1())
        
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
        self.ocp.solver_options.tol = 1e-5
        self.ocp.solver_options.print_level = 0

        self.ocp.solver_options.qp_solver_iter_max = 200
        self.ocp.qp_solver_warm_start = 2
        self.ocp.solver_options.hpipm_mode = 'SPEED'
        if problem.model_external_shared_lib_dir is not None and problem.model_external_shared_lib_name is not None:
            self.ocp.solver_options.model_external_shared_lib_dir = problem.model_external_shared_lib_dir
            self.ocp.solver_options.model_external_shared_lib_name = problem.model_external_shared_lib_name + ' -l' + problem.model_external_shared_lib_name
        
        if problem.has_neural_cost:
            self.ocp.solver_options.hessian_approx = "EXACT"
        else:
            self.ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        self.ocp.solver_options.levenberg_marquardt = 1e-2
        self.ocp.solver_options.nlp_solver_step_length = 0.75

        self.acados_solver = AcadosOcpSolver(self.ocp, json_file="acados_ocp.json")

    def _shift_horizon(self):
        self.xstar[:-1] = self.xstar[1:]
        self.ustar[:-1] = self.ustar[1:]

    def solve(self, x_initial: np.ndarray, params: List[np.ndarray]):
        self.acados_solver.set(0, "lbx", x_initial)
        self.acados_solver.set(0, "ubx", x_initial)

        if self.problem.get_terminal_cost_params_fun is not None:
            terminal_cost_input = self.problem.ocp_x_to_terminal_state_fun(self.xstar[-1, :], params[-1]).full().T
            terminal_cost_params = self.problem.get_terminal_cost_params_fun(terminal_cost_input)[0]
            params[-1] = np.r_[params[-1], terminal_cost_params]

        # Fill in x0, u0 and p for the solver:
        for i in range(self.N + 1):
            self.acados_solver.set(i, "x", self.xstar[i])
            if i < self.N:
                self.acados_solver.set(i, "u", self.ustar[i])
                self.acados_solver.set(i, "p", np.r_[params[i], np.zeros(self.n_param_pad)])
            else:
                self.acados_solver.set(i, "p", np.r_[np.zeros(self.n_terminal_param_pad), params[i]])

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
        else:
            print("STATUS", status)
            print(self.acados_solver.get_cost())
            print(self.acados_solver.get_residuals())
            print()

        state_horizon = self.xstar.copy()
        control_horizon = self.ustar.copy()
        self._shift_horizon()

        return state_horizon, control_horizon, status in [0, 2]