import numpy as np
import casadi as cs
import scipy
from acados_template import AcadosOcp, AcadosOcpSolver
from mpc_solvers.mpc_problem import SymbolicMPCProblem


class AcadosSolver:
    def __init__(self, problem: SymbolicMPCProblem):
        self.N = problem.N
        
        self.sample_time = problem.h
        self.time_steps = np.tile(self.sample_time, self.N+1)
        self.tf = self.time_steps.sum()

        self.n_controls = 0
        self.n_states = 0
        
        self.ocp = None
        self.acados_solver = None
        self._create_solver(problem)
        self.initialized = False

        self.x0 = np.zeros([self.N+1, self.n_states])
        self.u0 = np.zeros([self.N, self.n_controls])

    def _create_solver(self, problem: SymbolicMPCProblem):
        self.ocp = AcadosOcp()

        self.ocp.model.name = "acados_ocp"
        self.ocp.dims.N = problem.N
        # States
        self.ocp.model.x = problem.ocp_x
        # Controls
        self.ocp.model.u = problem.ocp_u
        # Parameters
        self.ocp.model.p = problem.ocp_p

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
        # ocp.constraints.lsbx = np.zeros(3)
        # ocp.constraints.usbx = np.zeros(3)
        # ocp.constraints.lsbx_e = ocp.constraints.lsbx
        # ocp.constraints.usbx_e = ocp.constraints.usbx
        # ocp.constraints.idxsbx = np.arange(start=3, stop=6)
        # ocp.constraints.idxsbx_e = ocp.constraints.idxsbx
        state_slack_weights = np.array([])

        # Control constraints:
        self.ocp.constraints.lbu = problem.lbu_vec
        self.ocp.constraints.ubu = problem.ubu_vec
        self.ocp.constraints.idxbu = np.arange(problem.ocp_u.size()[0])

        # Nonlinear constraints:
        if problem.g_fun is not None:
            self.ocp.model.con_h_expr = problem.g_fun(problem.ocp_x, problem.ocp_u, problem.ocp_p)
            self.ocp.constraints.idxsh = np.arange(0)
            nonlinear_slack_weights = np.array([])
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
        self.ocp.cost.W = scipy.linalg.block_diag(np.ones(self.ocp.model.cost_y_expr.size1()))

        # Terminal cost
        if problem._l4c_model is not None:
            self.ocp.cost.cost_type_e = 'EXTERNAL'
            neural_cost_state = problem.ocp_x_to_terminal_state_fun()
        else:
            self.ocp.cost.cost_type = 'NONLINEAR_LS'
            self.ocp.model.cost_y_expr_e = problem.cost_fun(problem.ocp_x, problem.ocp_u, problem.ocp_p)
            self.ocp.cost.yref_e = np.zeros(self.ocp.model.cost_y_expr.size1()) 
            self.ocp.cost.W_e = scipy.linalg.block_diag(np.ones(self.ocp.model.cost_y_expr.size1()))
        
        # Initialize initial conditions:
        self.n_states = self.ocp.model.x.size()[0]
        self.n_controls = self.ocp.model.u.size()[0]
        self.n_parameters = self.ocp.model.p.size()[0]
        self.ocp.constraints.x0 = np.zeros(self.n_states)
        self.ocp.parameter_values = np.zeros(self.n_parameters)

        # Integrator settings:
        self.ocp.solver_options.time_steps = self.time_steps
        self.ocp.solver_options.tf = self.tf
        self.ocp.solver_options.integrator_type = "DISCRETE"

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
            self.ocp.solver_options.model_external_shared_lib_dir = self.l4c_model.shared_lib_dir
            self.ocp.solver_options.model_external_shared_lib_name = self.l4c_model.name + ' -l' + self.l4c_model.name
            self.ocp.solver_options.hessian_approx = "EXACT"
        else:
            self.ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        self.ocp.solver_options.levenberg_marquardt = 1e-4
        # self.ocp.solver_options.nlp_solver_step_length = 0.75

        self.acados_solver = AcadosOcpSolver(self.ocp, json_file="acados_ocp.json")

    def initialize(self, initial_state, kappa):
        self.acados_solver.set(0, "lbx", initial_state)
        self.acados_solver.set(0, "ubx", initial_state)

        if self.initialized:
            self.x0[0] = initial_state
        else:
            self.x0 = np.tile(initial_state, (self.N+1, 1))

        if self.l4c_model is not None and self.linearize_cost_model:
            # Replicate the neural cost model inputs for linearization:
            kappa_values = np.tile(kappa, (self.N+1, 1))
            neural_states = np.c_[self.x0[:, :2], np.sin(self.x0[:, 2]), np.cos(self.x0[:, 2]), 
                                  kappa_values[:, :2], np.sin(kappa_values[:, 2]), np.cos(kappa_values[:, 2]), 
                                  self.x0[:, 3:], 
                                  np.r_[self.u0 / [config.avant_max_dot_beta, config.avant_max_a], np.zeros([1, 2])]]
            print(neural_states.shape)
            neural_cost_params = self.l4c_model.get_params(neural_states)

        # Fill in x0, u0 and p for the solver:
        for i in range(self.N + 1):
            if self.l4c_model is not None and self.linearize_cost_model:
                self.acados_solver.set(i, "p", np.r_[kappa, self.time_steps[i], neural_cost_params[i]])
            else:
                self.acados_solver.set(i, "p", np.r_[kappa, self.time_steps[i]])
            self.acados_solver.set(i, "x", self.x0[i])
            if i < self.N:
                self.acados_solver.set(i, "u", self.u0[i])

    def _shift_horizon(self):
        self.x0[:-1] = self.x0[1:]
        self.u0[:-1] = self.u0[1:]

    def solve(self):
        for i in range(25):
            status = self.acados_solver.solve()

        if status in [0, 2]:   # Success or timeout
            self.initialized = True
            for i in range(self.N+1):
                x = self.acados_solver.get(i, "x")
                self.x0[i] = x
                if i < self.N:
                    u = self.acados_solver.get(i, "u")
                    self.u0[i] = u
        else:
            print("STATUS", status)
            print(self.acados_solver.get_cost())
            print(self.acados_solver.get_residuals())
            print()

        state_horizon = self.x0.copy()
        control_horizon = self.u0.copy()
        self._shift_horizon()

        return state_horizon, control_horizon, status in [0, 2]