import config
import scipy
import numpy as np
import json
from casadi import *
from acados_template import AcadosOcp, AcadosOcpSolver

fake_inf = 1e7

class Solver:
    def __init__(self, N, sample_time, sample_time2=None):
        if sample_time2 is None:
            sample_time2 = sample_time
        
        self.N = N
        self.sample_time = sample_time
        self.time_steps = np.linspace(sample_time, sample_time2, N+1)
        
        self.tf = self.time_steps.sum()
        self.n_controls = 0
        self.n_states = 0

        with open('/home/aleksi/thesis_ws/src/preference_rewards/avant_parameters/ensemble_mean_parameters.json', 'r') as infile:
            self.params = json.load(infile)

        self.ocp = None
        self.acados_solver = None
        self._create_solver()

        self.initialized = False
        self.x0 = np.zeros([self.N+1, self.n_states])
        self.u0 = np.zeros([self.N, self.n_controls])


    def _create_solver(self):
        self.ocp = AcadosOcp()
        self.ocp.model.name = "avant"
        self.ocp.dims.N = self.N

        # States
        x_f = MX.sym("x_f")
        y_f = MX.sym("y_f")
        theta_f = MX.sym("theta_f")
        beta = MX.sym("beta")
        dot_beta = MX.sym("dot_beta")
        v = MX.sym("v")
        self.ocp.model.x = vertcat(x_f, y_f, theta_f, beta, dot_beta, v)

        # Controls
        dot_dot_beta = MX.sym("dot_dot_beta")
        a = MX.sym("a")
        self.ocp.model.u = vertcat(dot_dot_beta, a)

        # Parameters
        x_goal = MX.sym("x_goal")
        y_goal = MX.sym("y_goal")
        theta_goal = MX.sym("y_goal")
        dt = MX.sym("dt")
        cost_scaler = MX.sym("cost_scaler")

        self.ocp.model.p = vertcat(x_goal, y_goal, theta_goal, dt, cost_scaler)

        # Continuous dynamics:
        alpha = pi + beta
        omega_f = v * config.avant_lf / tan(alpha/2)
        f_expr = vertcat(
            v * cos(theta_f),
            v * sin(theta_f),
            omega_f,
            dot_beta,
            dot_dot_beta,
            a
        )
        f = Function('f', [self.ocp.model.x, self.ocp.model.u], [f_expr])
        # Runge Kutta 4 discrete dynamics:
        k1 = f(self.ocp.model.x, self.ocp.model.u)
        k2 = f(self.ocp.model.x + dt / 2 * k1, self.ocp.model.u)
        k3 = f(self.ocp.model.x + dt / 2 * k2, self.ocp.model.u)
        k4 = f(self.ocp.model.x + dt * k3, self.ocp.model.u)
        dynamics = dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        self.ocp.model.disc_dyn_expr = self.ocp.model.x + dynamics

        # State constraints
        self.ocp.constraints.lbx = np.array([
            -fake_inf, -fake_inf, -fake_inf, -config.avant_max_beta, -config.avant_max_dot_beta,
            config.avant_min_v
        ])
        self.ocp.constraints.ubx = np.array([
            fake_inf, fake_inf, fake_inf, config.avant_max_beta, config.avant_max_dot_beta, 
            config.avant_max_v
        ])
        self.ocp.constraints.lbx_e = self.ocp.constraints.lbx
        self.ocp.constraints.ubx_e = self.ocp.constraints.ubx
        self.ocp.constraints.idxbx = np.arange(self.ocp.model.x.size()[0])
        self.ocp.constraints.idxbx_e = self.ocp.constraints.idxbx

        self.ocp.constraints.lsbx = np.zeros(3)
        self.ocp.constraints.usbx = np.zeros(3)
        self.ocp.constraints.lsbx_e = self.ocp.constraints.lsbx
        self.ocp.constraints.usbx_e = self.ocp.constraints.usbx
        self.ocp.constraints.idxsbx = np.arange(start=3, stop=6)
        self.ocp.constraints.idxsbx_e = self.ocp.constraints.idxsbx
        state_slack_weights = np.array([1000, 10, 10])

        # Control constraints:
        self.ocp.constraints.lbu = np.array([
            -config.avant_max_dot_dot_beta, -config.max_a
        ])
        self.ocp.constraints.ubu = np.array([
            config.avant_max_dot_dot_beta, config.max_a,
        ])
        self.ocp.constraints.idxbu = np.arange(self.ocp.model.u.size()[0])

        # Nonlinear constraints:
        self.ocp.constraints.lh = np.array([3])
        self.ocp.constraints.uh = np.array([fake_inf])
        self.ocp.constraints.lh_e = self.ocp.constraints.lh
        self.ocp.constraints.uh_e = self.ocp.constraints.uh
        front_circle_x = x_goal + 1.5*cos(theta_goal)
        front_circle_y = y_goal + 1.5*sin(theta_goal)
        front_circle_dist = (front_circle_x - x_f)**2 + (front_circle_y - y_f)**2
        self.ocp.model.con_h_expr = vertcat(
            front_circle_dist
        )
        self.ocp.model.con_h_expr_e = self.ocp.model.con_h_expr
        self.ocp.constraints.idxsh = np.arange(1)
        self.ocp.constraints.idxsh_e = np.arange(1)
        nonlinear_slack_weights = np.array([1000])
        nonlinear_slack_weights_e = nonlinear_slack_weights

        self.n_states = self.ocp.model.x.size()[0]
        self.n_controls = self.ocp.model.u.size()[0]
        self.n_parameters = self.ocp.model.p.size()[0]

        self.ocp.cost.zl =   np.r_[state_slack_weights, nonlinear_slack_weights]
        self.ocp.cost.zl_e = np.r_[state_slack_weights, nonlinear_slack_weights_e]
        self.ocp.cost.zu =   np.r_[state_slack_weights, nonlinear_slack_weights]
        self.ocp.cost.zu_e = np.r_[state_slack_weights, nonlinear_slack_weights_e]

        self.ocp.cost.Zl =   np.r_[state_slack_weights, nonlinear_slack_weights]
        self.ocp.cost.Zl_e = np.r_[state_slack_weights, nonlinear_slack_weights_e]
        self.ocp.cost.Zu =   np.r_[state_slack_weights, nonlinear_slack_weights]
        self.ocp.cost.Zu_e = np.r_[state_slack_weights, nonlinear_slack_weights_e]

        # Nonlinear LS cost:
        self.ocp.cost.cost_type = 'NONLINEAR_LS'
        self.ocp.cost.cost_type_e = 'NONLINEAR_LS'

        goal_dist = ((x_f - x_goal)**2 + (y_f - y_goal)**2)
        goal_heading = self.params["heading_error_magnitude"] * (1 - cos(theta_goal - theta_f)) # 180/pi * atan2(sin(theta_goal - theta_f), cos(theta_goal - theta_f))
        radius_scaler = exp(-(goal_dist/self.params["scaling_radius"])**2)

        perpendicular_dist = (x_f - x_goal) * cos(theta_goal) + (y_f - y_goal) * sin(theta_goal)
        e_perp = self.params["perpendicular_error_magnitude"] * (
            tanh((perpendicular_dist - self.params["perpendicular_error_shift"]) * self.params["perpendicular_error_scaler"]) + 1
        )

        self.ocp.model.cost_y_expr = cost_scaler * vertcat(
            self.params["goal_dist_weight"]*(x_f - x_goal), self.params["goal_dist_weight"]*(y_f - y_goal), 
            radius_scaler * goal_heading, e_perp,
            a, dot_dot_beta * 180/pi, 
            v, radius_scaler * 180/pi * dot_beta
        )
        self.ocp.model.cost_y_expr_e = vertcat(
            self.params["goal_dist_weight"]*(x_f - x_goal), self.params["goal_dist_weight"]*(y_f - y_goal), radius_scaler * goal_heading
        )
        self.ocp.cost.yref = np.zeros(8)
        self.ocp.cost.yref_e = np.zeros(3)
        self.ocp.cost.W = scipy.linalg.block_diag(
            1, 1, 
            1, 1,
            1e-2, 1e-3, 
            1e-2, 1e-3)
        self.ocp.cost.W_e = scipy.linalg.block_diag(1, 1, 1)

        # Initialize initial conditions:
        self.ocp.constraints.x0 = np.zeros(self.n_states)
        self.ocp.parameter_values = np.zeros(self.n_parameters)

        self.ocp.solver_options.time_steps = self.time_steps
        self.ocp.solver_options.tf = self.tf
        self.ocp.solver_options.integrator_type = "DISCRETE"

        self.ocp.solver_options.nlp_solver_type = "SQP_RTI"
        self.ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
        self.ocp.solver_options.qp_solver_iter_max = 200
        self.ocp.qp_solver_warm_start = 1
        #self.ocp.solver_options.hpipm_mode = 'ROBUST'
        self.ocp.solver_options.hpipm_mode = 'SPEED'
        self.ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        self.ocp.solver_options.regularize_method = "PROJECT"
        self.ocp.solver_options.levenberg_marquardt = 1e-2
        # self.ocp.solver_options.nlp_solver_step_length = 0.75
        self.ocp.solver_options.tol = 1e-5
        self.ocp.solver_options.print_level = 0

        self.acados_solver = AcadosOcpSolver(self.ocp, json_file="acados_ocp.json")

    def initialize(self, initial_state, kappa_values):
        self.acados_solver.set(0, "lbx", initial_state)
        self.acados_solver.set(0, "ubx", initial_state)

        if self.initialized:
            self.x0[0] = initial_state
        else:
            self.x0 = np.tile(initial_state, (self.N+1, 1))

        discounts = (1 - np.power(self.params["discount_power"], -self.params["discount_scaler"] * np.arange(self.N+1)))

        # Fill in x0, u0 and p for the solver:
        for i in range(self.N + 1):
            self.acados_solver.set(i, "p", np.r_[kappa_values, self.time_steps[i], discounts[i]])
            self.acados_solver.set(i, "x", self.x0[i])
            if i < self.N:
                self.acados_solver.set(i, "u", self.u0[i])

    def _shift_horizon(self):
        self.x0[:-1] = self.x0[1:]
        self.u0[:-1] = self.u0[1:]

    def solve(self):
        for i in range(15):
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