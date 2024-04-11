import numpy as np
import casadi as cs
import pygame
import torch
import config
from sparse_to_dense_reward.avant_goal_env import AvantGoalEnv
from mpc_solvers.mpc_problem import SymbolicMPCProblem
from mpc_solvers.casadi_solver import CasadiSolver


class MPCActor:
    def __init__(self, linearize=True):
        super().__init__()

        fake_inf = 1e7
        
        # States
        x_f = cs.MX.sym("x_f")
        y_f = cs.MX.sym("y_f")
        theta_f = cs.MX.sym("theta_f")
        beta = cs.MX.sym("beta")
        dot_beta = cs.MX.sym("dot_beta")
        v = cs.MX.sym("v")
        ocp_x = cs.vertcat(x_f, y_f, theta_f, beta, dot_beta, v)
        lbx_vec = np.array([
            -fake_inf, -fake_inf, -fake_inf, -config.avant_max_beta, -config.avant_max_dot_beta,
            config.avant_min_v
        ])
        ubx_vec = np.array([
            fake_inf, fake_inf, fake_inf, config.avant_max_beta, config.avant_max_dot_beta, 
            config.avant_max_v
        ])

        # Controls
        dot_dot_beta = cs.MX.sym("dot_dot_beta")
        a = cs.MX.sym("a")
        ocp_u = cs.vertcat(dot_dot_beta, a)
        lbu_vec = np.array([
            -config.avant_max_dot_dot_beta, -config.avant_max_a
        ])
        ubu_vec = np.array([
            config.avant_max_dot_dot_beta, config.avant_max_a,
        ])

        # Params
        x_goal = cs.MX.sym("x_goal")
        y_goal = cs.MX.sym("y_goal")
        theta_goal = cs.MX.sym("y_goal")
        ocp_p = cs.vertcat()
        terminal_ocp_p = cs.vertcat(x_goal, y_goal, theta_goal)

        # Continuous dynamics:
        alpha = cs.pi + beta
        omega_f = v * config.avant_lf / cs.tan(alpha/2)
        f_expr = cs.vertcat(
            v * cs.cos(theta_f),
            v * cs.sin(theta_f),
            omega_f,
            dot_beta,
            dot_dot_beta,
            a
        )
        f = cs.Function('f', [ocp_x, ocp_u, ocp_p], [f_expr])

        # Stage cost:
        l = cs.Function('l', [ocp_x, ocp_u, ocp_p], [0])       

        problem = SymbolicMPCProblem(
            N=1,
            h=0.1,
            ocp_x=ocp_x,
            lbx_vec=lbx_vec,
            ubx_vec=ubx_vec,
            ocp_u=ocp_u,
            lbu_vec=lbu_vec,
            ubu_vec=ubu_vec,
            ocp_p=ocp_p,
            terminal_ocp_p=terminal_ocp_p,
            dynamics_fun=f, 
            cost_fun=l
        )
 
        # Terminal cost with Q network:
        neural_cost_state = cs.vertcat(x_f,    y_f,    cs.sin(theta_f),    cs.cos(theta_f), 
                                       x_goal, y_goal, cs.sin(theta_goal), cs.cos(theta_goal), 
                                       beta, dot_beta, v, 
                                       0, 0)
        model = torch.load("avant_critic")
        problem.add_terminal_neural_cost(model=model, model_state=neural_cost_state, linearize=linearize)

        self.solver = CasadiSolver(problem)

    def act(self, observation) -> np.ndarray:
        obs = observation["observation"][0]
        achieved_goal = observation["desired_goal"][0]
        achieved_goal = np.r_[achieved_goal[:2], np.arctan2(achieved_goal[2], achieved_goal[3])]
        obs = np.r_[achieved_goal, obs]

        desired_goal = observation["desired_goal"][0]
        desired_goal = np.r_[desired_goal[:2], np.arctan2(desired_goal[2], desired_goal[3])]
        params = [np.empty(0) for _ in range(self.solver.N + 1)]
        params[-1] = desired_goal
        
        sol_x, sol_u, sol = self.solver.solve(obs, params)
        print(sol_u)
        
        return sol_u[0]

# Initialize environment and actor
env = AvantGoalEnv(num_envs=1, dt=0.1, time_limit_s=20, device='cpu')
actor = MPCActor(linearize=True)

screen = pygame.display.set_mode([env.RENDER_RESOLUTION, env.RENDER_RESOLUTION])
clock = pygame.time.Clock()

# Frame rate and corresponding real-time frame duration
frame_rate = 30
normal_frame_duration_ms = 1000 // frame_rate
# Adjust frame duration for 0.25x real-time speed
slow_motion_multiplier = 4  # Slow down by this factor
frame_duration_ms = normal_frame_duration_ms * slow_motion_multiplier

# Main simulation loop
obs = env.reset()
while True:
    start_ticks = pygame.time.get_ticks()  # Get the start time of the current frame

    action = actor.act(obs) / env.dynamics.control_scalers.cpu().numpy()
    obs, reward, done, _ = env.step(action.reshape(1, -1).astype(np.float32))
    frame = env.render(mode='rgb_array')
    pygame.surfarray.blit_array(screen, frame.transpose(1, 0, 2))
    pygame.display.flip()

    # Ensure each frame lasts long enough to simulate 0.25x speed
    elapsed = pygame.time.get_ticks() - start_ticks
    if elapsed < frame_duration_ms:
        pygame.time.delay(frame_duration_ms - elapsed)