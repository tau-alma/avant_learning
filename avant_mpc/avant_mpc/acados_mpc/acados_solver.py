import json
import config
import torch
import l4casadi as l4c
import numpy as np
from acados_template import AcadosOcp, AcadosOcpSolver
from acados_mpc.ocp import build_ocp

class AcadosSolver:
    def __init__(self, N, sample_time, sample_time2=None, cost_model_path=None, linearize_cost_model=False):
        self.N = N
        
        self.sample_time = sample_time
        if sample_time2 is None:
            sample_time2 = sample_time
        self.time_steps = np.linspace(sample_time, sample_time2, N+1)
        self.tf = self.time_steps.sum()

        self.n_controls = 0
        self.n_states = 0
        
        self.linearize_cost_model = linearize_cost_model
        self.l4c_model = None
        if cost_model_path is not None:
            model = torch.load(cost_model_path).eval()
            if not linearize_cost_model:
                self.l4c_model = l4c.l4casadi.L4CasADi(model, device="cpu")
            else:
                self.l4c_model = l4c.realtime.RealTimeL4CasADi(model, device="cpu", approximation_order=2)

        self.ocp = None
        self.acados_solver = None
        self._create_solver()
        self.initialized = False

        self.x0 = np.zeros([self.N+1, self.n_states])
        self.u0 = np.zeros([self.N, self.n_controls])

    def _create_solver(self):
        self.ocp = AcadosOcp()

        build_ocp(self.ocp, self.N, self.l4c_model, self.linearize_cost_model)
        
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

        if self.l4c_model is not None:
            self.ocp.solver_options.qp_solver_iter_max = 1000
            self.ocp.qp_solver_warm_start = 0
            self.ocp.solver_options.hpipm_mode = 'BALANCE'
            self.ocp.solver_options.hessian_approx = "EXACT"
            self.ocp.solver_options.levenberg_marquardt = 0.
            self.ocp.solver_options.nlp_solver_step_length = 0.05
            self.ocp.solver_options.globalization_use_SOC = 1
            # self.ocp.solver_options.line_search_use_sufficient_descent = 1
            # self.ocp.solver_options.qp_solver_ric_alg = 1
        else:
            self.ocp.solver_options.qp_solver_iter_max = 200
            self.ocp.qp_solver_warm_start = 2
            self.ocp.solver_options.hpipm_mode = 'SPEED'
            self.ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
            self.ocp.solver_options.levenberg_marquardt = 1e-4
            # self.ocp.solver_options.nlp_solver_step_length = 0.75

        # L4casadi setup:
        if self.l4c_model is not None and not self.linearize_cost_model:
            self.ocp.solver_options.model_external_shared_lib_dir = self.l4c_model.shared_lib_dir
            self.ocp.solver_options.model_external_shared_lib_name = self.l4c_model.name + ' -l' + self.l4c_model.name

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
        for i in range(50):
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