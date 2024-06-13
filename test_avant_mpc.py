import numpy as np
import casadi as cs
import pygame
import torch
import time
import config
from argparse import ArgumentParser
from sparse_to_dense_reward.avant_goal_env import AvantGoalEnv
from mpc_solvers.mpc_problem import SymbolicMPCProblem
from mpc_solvers.casadi_collocation_solver import CasadiCollocationSolver
from mpc_solvers.casadi_shooting_solver import CasadiShootingSolver
from mpc_solvers.acados_solver import AcadosSolver


class MPCActor:
    def __init__(self, solver_class, linearize=False):
        self.history = []
        self.last_steer = 0
        super().__init__()

        fake_inf = 1e7
        
        # States
        x_f = cs.MX.sym("x_f")
        y_f = cs.MX.sym("y_f")
        theta_f = cs.MX.sym("theta_f")
        beta = cs.MX.sym("beta")
        dot_beta = cs.MX.sym("dot_beta")
        v_f_ref = cs.MX.sym("v_f_ref")
        steer = cs.MX.sym("steer")

        ocp_x = cs.vertcat(x_f, y_f, theta_f, beta, dot_beta, v_f_ref, steer)
        lbx_vec = np.array([
            -fake_inf, -fake_inf, -fake_inf, -config.avant_max_beta, -config.avant_max_dot_beta,
            config.avant_min_v, -1
        ])
        ubx_vec = np.array([
            fake_inf, fake_inf, fake_inf, config.avant_max_beta, config.avant_max_dot_beta, 
            config.avant_max_v, 1
        ])
        ocp_x_slacks = {3: 100, 4: 100, 5: 100}

        # Controls
        dot_steer = cs.MX.sym("dot_steer")
        a_f = cs.MX.sym("a_f")
        ocp_u = cs.vertcat(dot_steer, a_f)
        lbu_vec = np.array([
            -1, -config.avant_max_a
        ])
        ubu_vec = np.array([
            1, config.avant_max_a,
        ])

        # Params
        x_goal = cs.MX.sym("x_goal")
        y_goal = cs.MX.sym("y_goal")
        theta_goal = cs.MX.sym("y_goal")
        t = cs.MX.sym("t")
        # ocp_p = cs.vertcat(x_goal, y_goal, theta_goal, t)
        ocp_p = cs.vertcat(t)
        terminal_ocp_p = cs.vertcat(x_goal, y_goal, theta_goal)

        # Continuous dynamics:
        omega_f = -(
            (config.avant_lr * dot_beta + v_f_ref * cs.sin(beta)) 
            / (config.avant_lf * cs.cos(beta) + config.avant_lr)
        )

        a = 0.127                  # AFS parameter, check the paper page(1) Figure 1: AFS mechanism
        b = 0.495                  # AFS parameter, check the paper page(1) Figure 1: AFS mechanism
        eps0 = 1.4049900478554351  # the angle from of the hydraulic sylinder check the paper page(1) Figure (1) 
        eps = eps0 - beta
        k = 10 * a * b * cs.sin(eps) / cs.sqrt(a**2 + b**2 - 2*a*b*cs.cos(eps))
        dot_dot_beta = (dot_steer * k) / k**2

        f_expr = cs.vertcat(
            v_f_ref * cs.cos(theta_f),
            v_f_ref * cs.sin(theta_f),
            omega_f,
            dot_beta,
            dot_dot_beta,
            a_f,
            dot_steer
        )
        f = cs.Function('f', [ocp_x, ocp_u, ocp_p], [f_expr])

        # Stage cost:
        l = cs.Function('l', [ocp_x, ocp_u, ocp_p], [0])  

        # For empirical cost experiments:
        goal_heading = (90 * (1 - cs.cos(theta_goal - theta_f)) )**2
        goal_scaled_dist = (1e1*(x_f - x_goal))**2 + (1e1*(y_f - y_goal))**2
        # l  = cs.Function('l', [ocp_x, ocp_u, ocp_p], [t*(dot_dot_beta**2 + a**2 + v**2)])           
        # l_t = cs.Function('l_t', [ocp_x, terminal_ocp_p], [goal_scaled_dist + goal_heading])

        problem = SymbolicMPCProblem(
            N=5,
            h=0.2,
            ocp_x=ocp_x,
            lbx_vec=lbx_vec,
            ubx_vec=ubx_vec,
            ocp_x_slacks=ocp_x_slacks,
            ocp_u=ocp_u,
            lbu_vec=lbu_vec,
            ubu_vec=ubu_vec,
            ocp_p=ocp_p,
            terminal_ocp_p=terminal_ocp_p,
            dynamics_fun=f, 
            cost_fun=l,
            #terminal_cost_fun=l_t
        )
 
        # Terminal cost with Q network:
        neural_cost_state = cs.vertcat(x_goal - x_f,       y_goal - y_f,    
                                       cs.sin(theta_f),    cs.cos(theta_f), 
                                       cs.sin(theta_goal), cs.cos(theta_goal), 
                                       beta, dot_beta, v_f_ref, 
                                       0, 0)
        model = torch.load("avant_critic").eval()
        problem.add_terminal_neural_cost(model=model, model_state=neural_cost_state, linearize=linearize)

        self.solver = solver_class(problem, rebuild=True)
    
    def reset(self):
        self.last_steer = 0

    def act(self, observation) -> np.ndarray:
        obs = observation["observation"][0][:3]
        achieved_goal = observation["achieved_goal"][0]
        achieved_goal = np.r_[achieved_goal[:2], np.arctan2(achieved_goal[2], achieved_goal[3])]
        obs = np.r_[achieved_goal, obs, self.last_steer]

        desired_goal = observation["desired_goal"][0]
        desired_goal = np.r_[desired_goal[:2], np.arctan2(desired_goal[2], desired_goal[3])]
        params = [np.empty(0) for _ in range(self.solver.N + 1)]
        params = [np.r_[j*0.2/(self.solver.N*0.2)] for j in range(self.solver.N + 1)]
        params[-1] = desired_goal
        
        t1 = time.time_ns()
        sol_x, sol_u, sol = self.solver.solve(obs, params)
        t2 = time.time_ns()
        self.history.append((t2-t1)/1e6)
        print(np.asarray(self.history).mean())

        controls = np.r_[sol_x[1, 6], sol_u[0, 0], sol_x[1, 5]/3.5] # steer, dot_steer and gas (~= v_f_ref/3.5)

        return controls, sol_x[2:, :4]

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--solver")
    parser.add_argument("--linearize", action="store_true")
    args = parser.parse_args()

    if args.solver == "casadi_collocation":
        solver_class = CasadiCollocationSolver
    elif args.solver == "casadi_shooting":
        solver_class = CasadiShootingSolver
    elif args.solver == "acados":
        solver_class = AcadosSolver
    else:
        raise ValueError(f"Unknown solver {args.solver}")
    
    # Initialize environment and actor
    env = AvantGoalEnv(num_envs=1, dt=0.1, time_limit_s=30, device='cpu', num_obstacles=0, eval=True)
    actor = MPCActor(solver_class=solver_class, linearize=args.linearize)

    screen = pygame.display.set_mode([env.RENDER_RESOLUTION, env.RENDER_RESOLUTION])

    # Frame rate and corresponding real-time frame duration
    frame_rate = 30
    normal_frame_duration_ms = 1000 // frame_rate
    # Adjust frame duration for 0.25x real-time speed
    slow_motion_multiplier = 2  # Slow down by this factor
    frame_duration_ms = normal_frame_duration_ms * slow_motion_multiplier

    # Main simulation loop
    obs = env.reset()

    while True:
        for _ in pygame.event.get():
            pass
        start_ticks = pygame.time.get_ticks()  # Get the start time of the current frame

        action, horizon = actor.act(obs)
        action /= env.dynamics.control_scalers.cpu().numpy()
        
        obs, reward, done, trunc = env.step(action.reshape(1, -1).astype(np.float32))

        if done or trunc:
            actor.reset()

        frame = env.render(mode='rgb_array', horizon=horizon)
        pygame.surfarray.blit_array(screen, frame.transpose(1, 0, 2))
        pygame.display.flip()

        # Ensure each frame lasts long enough to simulate 0.25x speed
        elapsed = pygame.time.get_ticks() - start_ticks
        if elapsed < frame_duration_ms:
            pygame.time.delay(frame_duration_ms - elapsed)