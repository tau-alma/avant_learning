from casadi import *
import numpy as np
import config
import scipy

fake_inf = 1e7

def build_ocp(ocp, N, l4c_model=None, linearize_cost_model=False):
    ocp.model.name = "avant"
    ocp.dims.N = N

    # States
    x_f = MX.sym("x_f")
    y_f = MX.sym("y_f")
    theta_f = MX.sym("theta_f")
    beta = MX.sym("beta")
    dot_beta = MX.sym("dot_beta")
    v = MX.sym("v")
    ocp.model.x = vertcat(x_f, y_f, theta_f, beta, dot_beta, v)

    # Controls
    dot_dot_beta = MX.sym("dot_dot_beta")
    a = MX.sym("a")
    ocp.model.u = vertcat(dot_dot_beta, a)

    # Parameters
    dt = MX.sym("dt")
    x_goal = MX.sym("x_goal")
    y_goal = MX.sym("y_goal")
    theta_goal = MX.sym("y_goal")
    ocp.model.p = vertcat(x_goal, y_goal, theta_goal, dt)

    if l4c_model is not None:
        # For neural cost
        sym_cost_input = MX.sym("sym_input", 13, 1)
        sym_cost_output = l4c_model(sym_cost_input)

        if linearize_cost_model:
            sym_cost_params = l4c_model.get_sym_params()
            casadi_cost_func = Function('model_rt_approx', [sym_cost_input, sym_cost_params], [sym_cost_output]) 
            ocp.model.p = vertcat(x_goal, y_goal, theta_goal, dt, sym_cost_params)
        else:
            casadi_cost_func = Function('model', [sym_cost_input], [sym_cost_output]) 

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
    f = Function('f', [ocp.model.x, ocp.model.u], [f_expr])
    # Runge Kutta 4 discrete dynamics:
    k1 = f(ocp.model.x, ocp.model.u)
    k2 = f(ocp.model.x + dt / 2 * k1, ocp.model.u)
    k3 = f(ocp.model.x + dt / 2 * k2, ocp.model.u)
    k4 = f(ocp.model.x + dt * k3, ocp.model.u)
    dynamics = dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    ocp.model.disc_dyn_expr = ocp.model.x + dynamics

    # State constraints
    ocp.constraints.lbx = np.array([
        -fake_inf, -fake_inf, -fake_inf, -config.avant_max_beta, -config.avant_max_dot_beta,
        config.avant_min_v
    ])
    ocp.constraints.ubx = np.array([
        fake_inf, fake_inf, fake_inf, config.avant_max_beta, config.avant_max_dot_beta, 
        config.avant_max_v
    ])
    ocp.constraints.lbx_e = ocp.constraints.lbx
    ocp.constraints.ubx_e = ocp.constraints.ubx
    ocp.constraints.idxbx = np.arange(ocp.model.x.size()[0])
    ocp.constraints.idxbx_e = ocp.constraints.idxbx

    ocp.constraints.lsbx = np.zeros(3)
    ocp.constraints.usbx = np.zeros(3)
    ocp.constraints.lsbx_e = ocp.constraints.lsbx
    ocp.constraints.usbx_e = ocp.constraints.usbx
    ocp.constraints.idxsbx = np.arange(start=3, stop=6)
    ocp.constraints.idxsbx_e = ocp.constraints.idxsbx
    state_slack_weights = np.array([1000, 10, 10])

    # Control constraints:
    ocp.constraints.lbu = np.array([
        -config.avant_max_dot_dot_beta, -config.avant_max_a
    ])
    ocp.constraints.ubu = np.array([
        config.avant_max_dot_dot_beta, config.avant_max_a,
    ])
    ocp.constraints.idxbu = np.arange(ocp.model.u.size()[0])

    # Nonlinear constraints:
    ocp.model.con_h_expr = vertcat(
    )
    ocp.model.con_h_expr_e = ocp.model.con_h_expr
    ocp.constraints.idxsh = np.arange(0)
    ocp.constraints.idxsh_e = np.arange(0)
    nonlinear_slack_weights = np.array([])
    nonlinear_slack_weights_e = nonlinear_slack_weights

    # Slack penalties:
    ocp.cost.zl =   np.r_[state_slack_weights, nonlinear_slack_weights]
    ocp.cost.zl_e = np.r_[state_slack_weights, nonlinear_slack_weights_e]
    ocp.cost.zu =   np.r_[state_slack_weights, nonlinear_slack_weights]
    ocp.cost.zu_e = np.r_[state_slack_weights, nonlinear_slack_weights_e]
    ocp.cost.Zl =   np.r_[state_slack_weights, nonlinear_slack_weights]
    ocp.cost.Zl_e = np.r_[state_slack_weights, nonlinear_slack_weights_e]
    ocp.cost.Zu =   np.r_[state_slack_weights, nonlinear_slack_weights]
    ocp.cost.Zu_e = np.r_[state_slack_weights, nonlinear_slack_weights_e]

    # External neural cost:
    ocp.cost.cost_type = 'NONLINEAR_LS'
    ocp.cost.yref = np.zeros(4) 
    ocp.cost.W = scipy.linalg.block_diag(
        1e-4, 1e-3, 
        1e-9, 1e-9)
    ocp.model.cost_y_expr = vertcat(
        a, dot_dot_beta * 180/pi, 
        v, 180/pi * dot_beta
    )

    if l4c_model is not None:
        ocp.cost.cost_type_e = 'EXTERNAL'
        neural_cost_state = vertcat(x_f,    y_f,    sin(theta_f),    cos(theta_f), 
                                    x_goal, y_goal, sin(theta_goal), cos(theta_goal), 
                                    beta, dot_beta, v, 
                                    0, 0)
        if linearize_cost_model:
            ocp.model.cost_expr_ext_cost_e = -casadi_cost_func(neural_cost_state, sym_cost_params)
        else:
            ocp.model.cost_expr_ext_cost_e = -casadi_cost_func(neural_cost_state)
    else:
        ocp.cost.cost_type = 'NONLINEAR_LS'
        
        goal_dist = ((x_f - x_goal)**2 + (y_f - y_goal)**2)
        radius_scaler = exp(-(goal_dist/2)**2)
        goal_heading = radius_scaler * 90 * (1 - cos(theta_goal - theta_f))

        ocp.cost.yref = np.zeros(3) 
        ocp.cost.W = scipy.linalg.block_diag(1, 1, 1e-1)
        ocp.model.cost_y_expr = vertcat(
            1e1*(x_f - x_goal), 1e1*(y_f - y_goal), goal_heading
        )