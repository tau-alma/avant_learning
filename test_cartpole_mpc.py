import numpy as np
import casadi as cs
import pygame
import torch
from sparse_to_dense_reward.cartpole_env import ContinuousCartpoleEnv
from mpc_solvers.mpc_problem import SymbolicMPCProblem
from mpc_solvers.casadi_solver import CasadiSolver


class MPCActor:
    def __init__(self, linearize=True):
        super().__init__()

        fake_inf = 1e7
        
        # States
        x = cs.MX.sym("x")
        theta = cs.MX.sym("theta")
        v = cs.MX.sym("v")
        omega = cs.MX.sym("omega")
        ocp_x = cs.vertcat(x, theta, v, omega)
        # State constraint bounds
        lbx_vec = np.array([-9, -fake_inf, -3, -np.deg2rad(45)])
        ubx_vec = np.array([9, fake_inf, 3, np.deg2rad(45)])

        # Controls
        a = cs.MX.sym("a")
        ocp_u = cs.vertcat(a)
        # Control constraint bounds
        lbu_vec = np.array([-9.8])
        ubu_vec = np.array([9.8])

        # Params
        ocp_p = cs.vertcat()
        terminal_ocp_p = cs.vertcat()

        # Continuous dynamics:
        g = 9.81        # Gravitational acceleration
        l = 1.5         # length of pole
        l_c = l / 2     # Pole's center of mass location
        m_p = 1         # Mass of pole
        m_c = 2         # Mass of cart
        f_c = 0.25      # Friction coeff between cart and ground
        common_numerator = g * cs.sin(theta) - a * cs.cos(theta)
        common_denominator = l_c * (4/3 - m_p * cs.cos(theta)**2 / (m_p + m_c))
        dot_omega = common_numerator / common_denominator
        ax = a + m_p * l_c * dot_omega * cs.cos(theta) / (m_p + m_c) - m_p * l_c * omega**2 * cs.sin(theta) / (m_p + m_c) - f_c*v
        f_expr = cs.vertcat(
            v,  
            omega, 
            ax, 
            dot_omega  
        )
        f = cs.Function('f', [ocp_x, ocp_u, ocp_p], [f_expr])

        # Stage cost:
        l = cs.Function('l', [ocp_x, ocp_u, ocp_p], [0])       

        problem = SymbolicMPCProblem(
            N=10,
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
        neural_cost_state = cs.vertcat(ocp_x, 0)
        model = torch.load("cart_critic")
        problem.add_terminal_neural_cost(model=model, model_state=neural_cost_state, linearize=linearize)

        self.solver = CasadiSolver(problem)

    def act(self, observation: np.ndarray) -> np.ndarray:
        sol_x, sol_u, sol = self.solver.solve(observation[0], [np.empty(0) for _ in range(self.solver.N + 1)])
        return sol_u[0]

# Initialize environment and actor
env = ContinuousCartpoleEnv(num_envs=1, dt=0.1, time_limit_s=10, device='cpu')
actor = MPCActor(linearize=False)

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