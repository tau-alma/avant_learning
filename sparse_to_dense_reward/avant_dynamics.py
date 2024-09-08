import torch
import numpy as np
import sparse_to_dense_reward.config as config
from avant_modeling.gp import GPModel

class AvantDynamics:
    # State indices:
    x_f_idx = 0
    y_f_idx = 1
    theta_f_idx = 2
    beta_idx = 3
    dot_beta_idx = 4
    v_f_idx = 5

    # Additional states for handling the actuator delays (only used during evaluation, i.e. in "_mpc_dynamics_fun")
    steer_del3_idx = 6
    steer_del2_idx = 7
    steer_del1_idx = 8
    throttle_del1_idx = 9

    # Control indices for RL:
    u_dot_dot_beta_idx = 0
    u_a_f_idx = 1

    # Control indices for MPC:
    u_steer_idx = 0
    u_throttle_idx = 1

    def __init__(self, dt: float, device: str, eval: bool, eval_input_delay: bool = False):
        assert not eval_input_delay or (eval_input_delay and dt == 1/10), "Current model assumes dt = 1/10hz to correctly represent actuator delay"

        self.dt = dt
        self.device = device
        self.eval = eval
        self.eval_input_delay = eval_input_delay

        # Define control normalization constants:
        if not eval:
            self.control_scalers = torch.tensor([config.avant_max_dot_dot_beta, config.avant_max_a], dtype=torch.float32).to(device)
        else:
            self.control_scalers = torch.ones(2, dtype=torch.float32).to(device)

        # Define state bounds:
        if not eval:
            self.lbx = torch.tensor([
                -torch.inf, -torch.inf, -torch.inf, 
                -config.avant_max_beta, -config.avant_max_dot_beta, 
                config.avant_min_v
            ], dtype=torch.float32).to(device)
            self.ubx = torch.tensor([
                torch.inf, torch.inf, torch.inf, 
                config.avant_max_beta, config.avant_max_dot_beta,
                config.avant_max_v
            ], dtype=torch.float32).to(device)
        else:
            if eval_input_delay:
                self.lbx = torch.tensor([
                    -torch.inf, -torch.inf, -torch.inf, 
                    -config.avant_max_beta, -config.avant_max_dot_beta, 
                    config.avant_min_v,
                    -1, -1, -1, -1
                ], dtype=torch.float32).to(device)
                self.ubx = torch.tensor([
                    torch.inf, torch.inf, torch.inf, 
                    config.avant_max_beta, config.avant_max_dot_beta,
                    config.avant_max_v,
                    1, 1, 1, 1
                ], dtype=torch.float32).to(device)
            else:
                self.lbx = torch.tensor([
                    -torch.inf, -torch.inf, -torch.inf, 
                    -config.avant_max_beta, -config.avant_max_dot_beta, 
                    config.avant_min_v
                ], dtype=torch.float32).to(device)
                self.ubx = torch.tensor([
                    torch.inf, torch.inf, torch.inf, 
                    config.avant_max_beta, config.avant_max_dot_beta,
                    config.avant_max_v
                ], dtype=torch.float32).to(device)

            self.gp_dict = {}
            for name in ["omega_f", "truncated_v_f", "truncated_dot_beta"]:
                data_x = torch.load(f"sparse_to_dense_reward/{name}/{name}_gp_inputs.pth").to(device)
                data_y = torch.load(f"sparse_to_dense_reward/{name}/{name}_gp_targets.pth").to(device)
                gp = GPModel(data_x, data_y, train_epochs=0, device=device).to(device)
                gp.load_state_dict(torch.load(f"sparse_to_dense_reward/{name}/{name}_gp_model.pth"))
                self.gp_dict[name] = gp

    # Simple kinematic model with direct control of accelerations, used to train RL policy
    def _rl_dynamics_fun(self, x_values: torch.Tensor, u_values: torch.Tensor, add_noise=True) -> torch.Tensor:
        def continuous_dynamics(x_values: torch.Tensor, u_values: torch.Tensor):
            u_values = self.control_scalers * u_values

            desired_beta_accel = u_values[:, self.u_dot_dot_beta_idx]
            desired_linear_accel = u_values[:, self.u_a_f_idx]

            if add_noise:
                # Add some noise to the resulting accelerations
                std_beta_accel = 1/2 * (0.5*config.avant_max_dot_dot_beta) / (config.avant_max_beta**2 + config.avant_max_dot_beta**2) * (x_values[:, self.beta_idx]**2 + x_values[:, self.dot_beta_idx]**2)
                desired_beta_accel += torch.normal(mean=0, std=std_beta_accel)
                std_linear_accel = 1/2 * (0.5*config.avant_max_a) / (config.avant_max_v**2) * (x_values[:, self.v_f_idx]**2)
                desired_linear_accel += torch.normal(mean=0, std=std_linear_accel)

            # Limit the resulting accelerations
            limited_beta_accel = torch.max(
                torch.min(
                    desired_beta_accel, 
                    config.avant_max_dot_dot_beta * torch.ones_like(desired_beta_accel).to(u_values.device)
                ), 
                -config.avant_max_dot_dot_beta * torch.ones_like(desired_beta_accel).to(u_values.device)
            )
            limited_linear_accel = torch.max(
                torch.min(
                    desired_linear_accel, 
                    config.avant_max_a * torch.ones_like(desired_linear_accel).to(u_values.device)
                ), 
                -config.avant_max_a * torch.ones_like(desired_linear_accel).to(u_values.device)
            )

            omega_f = -(
                (config.avant_lr * x_values[:, self.dot_beta_idx] + x_values[:, self.v_f_idx] * torch.sin(x_values[:, self.beta_idx])) 
                / (config.avant_lf * torch.cos(x_values[:, self.beta_idx]) + config.avant_lr)
            )
            dot_state = torch.vstack([
                x_values[:, self.v_f_idx] * torch.cos(x_values[:, self.theta_f_idx]),
                x_values[:, self.v_f_idx] * torch.sin(x_values[:, self.theta_f_idx]),
                omega_f,
                x_values[:, self.dot_beta_idx],
                limited_beta_accel,
                limited_linear_accel
            ]).T
            return dot_state
        
        k1 = continuous_dynamics(x_values, u_values)
        k2 = continuous_dynamics(x_values + self.dt / 2 * k1, u_values)
        k3 = continuous_dynamics(x_values + self.dt / 2 * k2, u_values)
        k4 = continuous_dynamics(x_values + self.dt * k3, u_values)
        state_delta = self.dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        next_state = x_values + state_delta
        clamped_next_state = torch.max(torch.min(next_state, self.ubx), self.lbx)

        # Zero out any center link velocity (going towards the closest joint limit direction) at joint limit
        zero_dot_beta = clamped_next_state.clone()
        zero_dot_beta[:, self.dot_beta_idx] = 0
        selector = ((clamped_next_state[:, self.beta_idx] == -config.avant_max_beta) & (clamped_next_state[:, self.dot_beta_idx] < 0)).unsqueeze(1).expand(-1, 6)
        clamped_next_state = torch.where(
            selector,
            zero_dot_beta,
            clamped_next_state
        )
        selector = ((clamped_next_state[:, self.beta_idx] == config.avant_max_beta) & (clamped_next_state[:, self.dot_beta_idx] > 0)).unsqueeze(1).expand(-1, 6)
        clamped_next_state = torch.where(
            selector,
            zero_dot_beta,
            clamped_next_state
        )

        return clamped_next_state

    # More detailed dynamical model used to evaluate the designed controller
    def _mpc_dynamics_fun(self, x_values: torch.Tensor, u_values: torch.Tensor) -> torch.Tensor:

        def continuous_dynamics(x_values: torch.Tensor, u_values: torch.Tensor):
            u_values = self.control_scalers * u_values

            gp_omega_f_inputs = torch.vstack([
                x_values[:, self.beta_idx], x_values[:, self.dot_beta_idx], x_values[:, self.v_f_idx]
            ]).T
            gp_omega_f = self.gp_dict["omega_f"](gp_omega_f_inputs).mean
            omega_f = gp_omega_f

            # Nominal and GP for v_f
            gp_v_f_inputs = torch.vstack([
                x_values[:, self.beta_idx], x_values[:, self.dot_beta_idx], u_values[:, self.u_throttle_idx]
            ]).T
            gp_v_f = self.gp_dict["truncated_v_f"](gp_v_f_inputs).mean
            v_f = gp_v_f

            # Nominal and GP for dot_beta        
            gp_dot_beta_inputs = torch.vstack([
                x_values[:, self.beta_idx], v_f, u_values[:, self.u_steer_idx]
            ]).T
            gp_dot_beta = self.gp_dict["truncated_dot_beta"](gp_dot_beta_inputs).mean
            dot_beta = gp_dot_beta

            dot_state = torch.vstack([
                v_f * torch.cos(x_values[:, self.theta_f_idx]),
                v_f * torch.sin(x_values[:, self.theta_f_idx]),
                omega_f,
                dot_beta,

                (dot_beta - x_values[:, self.dot_beta_idx]) / self.dt,                              # dot_beta
                (v_f - x_values[:, self.v_f_idx]) / self.dt,                                        # a_f
            ]).T
            return dot_state

        def delayed_continuous_dynamics(x_values: torch.Tensor, u_values: torch.Tensor):
            u_values = self.control_scalers * u_values

            gp_omega_f_inputs = torch.vstack([
                x_values[:, self.beta_idx], x_values[:, self.dot_beta_idx], x_values[:, self.v_f_idx]
            ]).T
            gp_omega_f = self.gp_dict["omega_f"](gp_omega_f_inputs).rsample(torch.Size([1]))
            omega_f = gp_omega_f

            gp_v_f_inputs = torch.vstack([
                x_values[:, self.beta_idx], x_values[:, self.dot_beta_idx], x_values[:, self.throttle_del1_idx]
            ]).T
            gp_v_f = self.gp_dict["truncated_v_f"](gp_v_f_inputs).rsample(torch.Size([1]))
            v_f = gp_v_f

            gp_dot_beta_inputs = torch.vstack([
                x_values[:, self.beta_idx], v_f, x_values[:, self.steer_del3_idx]
            ]).T
            gp_dot_beta = self.gp_dict["truncated_dot_beta"](gp_dot_beta_inputs).rsample(torch.Size([1]))
            dot_beta = gp_dot_beta

            dot_state = torch.vstack([
                v_f * torch.cos(x_values[:, self.theta_f_idx]),
                v_f * torch.sin(x_values[:, self.theta_f_idx]),
                omega_f,
                dot_beta,

                (dot_beta - x_values[:, self.dot_beta_idx]) / self.dt,                              # dot_beta
                (v_f - x_values[:, self.v_f_idx]) / self.dt,                                        # a_f

                (x_values[:, self.steer_del2_idx] - x_values[:, self.steer_del3_idx]) / self.dt,    # del_steer3
                (x_values[:, self.steer_del1_idx] - x_values[:, self.steer_del2_idx]) / self.dt,    # del_steer2
                (u_values[:, self.u_steer_idx] - x_values[:, self.steer_del1_idx]) / self.dt,       # del_steer1

                (u_values[:, self.u_throttle_idx] - x_values[:, self.throttle_del1_idx]) / self.dt  # del_throttle1
            ]).T
            return dot_state

        if self.eval_input_delay:
            state_delta = self.dt * delayed_continuous_dynamics(x_values, u_values)
        else:
            state_delta = self.dt * continuous_dynamics(x_values, u_values)
        next_state = x_values + state_delta

        # Zero out any center link velocity (going towards the closest joint limit direction) at joint limit
        clamped_next_state = torch.max(torch.min(next_state, self.ubx), self.lbx)
        zero_dot_beta = clamped_next_state.clone()
        zero_dot_beta[:, self.dot_beta_idx] = 0
        clamped_next_state = torch.where(
            (clamped_next_state[:, self.beta_idx] == -config.avant_max_beta) & (clamped_next_state[:, self.dot_beta_idx] < 0),
            zero_dot_beta,
            clamped_next_state
        )
        clamped_next_state = torch.where(
            (clamped_next_state[:, self.beta_idx] == config.avant_max_beta) & (clamped_next_state[:, self.dot_beta_idx] > 0),
            zero_dot_beta,
            clamped_next_state
        )

        return clamped_next_state 

    def discrete_dynamics_fun(self, x_values: torch.Tensor, u_values: torch.Tensor) -> torch.Tensor:
        if not self.eval:
            return self._rl_dynamics_fun(x_values, u_values, add_noise=False)
        else:
            with torch.no_grad():
                return self._mpc_dynamics_fun(x_values, u_values)