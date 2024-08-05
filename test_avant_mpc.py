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
from utils import MLP

class MPCActor:
    def __init__(self, solver_class):
        super().__init__()
        self.history = []
        fake_inf = 1e7
        
        # States
        x_f = cs.MX.sym("x_f")
        y_f = cs.MX.sym("y_f")
        theta_f = cs.MX.sym("theta_f")
        beta = cs.MX.sym("beta")
        dot_beta_ref = cs.MX.sym("dot_beta_ref")
        v_f_ref = cs.MX.sym("v_f_ref")

        ocp_x = cs.vertcat(x_f, y_f, theta_f, beta, dot_beta_ref, v_f_ref)
        lbx_vec = np.array([
            -fake_inf, -fake_inf, -fake_inf, -config.avant_max_beta, -config.avant_max_dot_beta,
            config.avant_min_v
        ])
        ubx_vec = np.array([
            fake_inf, fake_inf, fake_inf, config.avant_max_beta, config.avant_max_dot_beta, 
            config.avant_max_v
        ])
        ocp_x_slacks = {3: 100, 4: 100, 5: 100}

        # Controls
        dot_dot_beta = cs.MX.sym("dot_dot_beta")
        a_f = cs.MX.sym("a_f")
        ocp_u = cs.vertcat(dot_dot_beta, a_f)
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
        policy_dot_beta = cs.MX.sym("policy_dot_beta")
        policy_v_f = cs.MX.sym("policy_v_f")
        ocp_p = cs.vertcat(x_goal, y_goal, theta_goal, policy_dot_beta, policy_v_f)

        # Continuous dynamics:
        omega_f = -(
            (config.avant_lr * dot_beta_ref + v_f_ref * cs.sin(beta)) 
            / (config.avant_lf * cs.cos(beta) + config.avant_lr)
        )
        f_expr = cs.vertcat(
            v_f_ref * cs.cos(theta_f),
            v_f_ref * cs.sin(theta_f),
            omega_f,
            dot_beta_ref,
            dot_dot_beta,
            a_f
        )
        f = cs.Function('f', [ocp_x, ocp_u, ocp_p], [f_expr])

        accel_penalty = 1e-3*dot_dot_beta**2 + 1e-3*a_f**2
        policy_deviation = (dot_beta_ref - policy_dot_beta)**2 + (v_f_ref - policy_v_f)**2
        standstill_turning_penalty = (1 - cs.tanh(v_f_ref**2 / 0.15)) * 10*(dot_beta_ref)**2
        l = cs.Function('l', [ocp_x, ocp_u, ocp_p], [accel_penalty + standstill_turning_penalty])

        problem = SymbolicMPCProblem(
            N=10,
            h=0.2,
            ocp_x=ocp_x,
            lbx_vec=lbx_vec,
            ubx_vec=ubx_vec,
            ocp_x_slacks=ocp_x_slacks,
            ocp_u=ocp_u,
            lbu_vec=lbu_vec,
            ubu_vec=ubu_vec,
            ocp_p=ocp_p,
            dynamics_fun=f, 
            stage_cost_fun=l
        )
 
        # Terminal cost with Q network:
        critic_state = cs.vertcat(
            x_goal - x_f,       y_goal - y_f,    
            cs.sin(theta_f),    cs.cos(theta_f), 
            cs.sin(theta_goal), cs.cos(theta_goal), 
            beta, dot_beta_ref, v_f_ref, 
            dot_beta_ref,       v_f_ref
        )
        critic_model = torch.load("avant_critic").eval()
        problem.add_stage_neural_cost(model=critic_model, model_state=critic_state)
        problem.add_terminal_neural_cost(model=critic_model, model_state=critic_state)

        # Reference targets with policy:
        policy_model = MLP(9, [18, 36, 72, 36, 18], 2).eval()
        policy_model.load_state_dict(torch.load("avant_actor"))
        actor_state = critic_state[:-2]
        sim_x = cs.vertcat(x_f, y_f, theta_f, beta)
        sim_u = cs.vertcat(dot_beta_ref, v_f_ref)
        sim_f_expr = cs.vertcat(
            v_f_ref * cs.cos(theta_f),
            v_f_ref * cs.sin(theta_f),
            omega_f,
            dot_beta_ref
        )
        f_s = cs.Function("sim_fun", [sim_x, sim_u], [sim_f_expr])
        # problem.add_neural_policy(model=policy_model, model_state=actor_state, sim_fun=f_s)

        self.solver = solver_class(problem, rebuild=True)

    def reset(self):
        pass            

    def act(self, observation) -> np.ndarray:
        obs = observation["observation"][0][:3]
        achieved_goal = observation["achieved_goal"][0]
        achieved_goal = np.r_[achieved_goal[:2], np.arctan2(achieved_goal[2], achieved_goal[3])]
        obs = np.r_[achieved_goal, obs]

        # Extract x, y, theta:
        desired_goal = observation["desired_goal"][0]
        desired_goal = np.r_[desired_goal[:2], np.arctan2(desired_goal[2], desired_goal[3])]
        params = [desired_goal for _ in range(self.solver.N + 1)]
        
        t1 = time.time_ns()
        sol_x, sol_u, sol = self.solver.solve(obs, params)
        t2 = time.time_ns()
        self.history.append((t2-t1)/1e6)
        print(np.asarray(self.history).mean())

        a = 0.127                  # AFS parameter, check the paper page(1) Figure 1: AFS mechanism
        b = 0.495                  # AFS parameter, check the paper page(1) Figure 1: AFS mechanism
        eps0 = 1.4049900478554351  # the angle from of the hydraulic sylinder check the paper page(1) Figure (1) 
        eps = eps0 - sol_x[1, 3]
        k = 10 * a * b * np.sin(eps) / np.sqrt(a**2 + b**2 - 2*a*b*np.cos(eps))
        u_steer = k * sol_x[1, 4]

        u_throttle = sol_x[1, 5] / 3
        controls = np.r_[u_steer, u_throttle]

        return controls, sol_x[2:, :4]
    
class RLActor:
    def __init__(self):
        self.network = torch.load("avant_actor_full").eval()
        self.beta_vel = 0
        self.vel_f = 0

    def reset(self):
        self.beta_vel = 0
        self.vel_f = 0

    def act(self, observation) -> np.ndarray:
        torch_state = {k: torch.from_numpy(v).to(self.network.device) for k, v in observation.items()}
        with torch.no_grad():
            policy_outputs = self.network(torch_state)[0]
        policy_outputs = policy_outputs.cpu().numpy()

        # Rate limit the change of beta velocity target command by max allowed acceleration:
        new_beta_vel = max(
            min(
                policy_outputs[0] * config.avant_max_dot_beta,
                self.beta_vel + 0.1 * config.avant_max_dot_dot_beta
            ),
            self.beta_vel - 0.1 * config.avant_max_dot_dot_beta
        )
        self.beta_vel = new_beta_vel

        # Rate limit the change of linear velocity target command by max allowed acceleration:
        self.vel_f = max(
            min(
                policy_outputs[1] * config.avant_max_v,
                self.vel_f + 0.1 * config.avant_max_a
            ),
            self.vel_f - 0.1 * config.avant_max_a
        )

        obs = observation["observation"][0][:3]
        a = 0.127                  # AFS parameter, check the paper page(1) Figure 1: AFS mechanism
        b = 0.495                  # AFS parameter, check the paper page(1) Figure 1: AFS mechanism
        eps0 = 1.4049900478554351  # the angle from of the hydraulic sylinder check the paper page(1) Figure (1) 
        eps = eps0 - obs[0]
        k = 10 * a * b * np.sin(eps) / np.sqrt(a**2 + b**2 - 2*a*b*np.cos(eps))

        env_controls = np.r_[k * self.beta_vel, self.vel_f/3]
        
        return env_controls, np.empty(0)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--solver")
    parser.add_argument("--rl", action="store_true")
    args = parser.parse_args()

    if not args.rl:
        if args.solver == "casadi_collocation":
            solver_class = CasadiCollocationSolver
        elif args.solver == "casadi_shooting":
            solver_class = CasadiShootingSolver
        elif args.solver == "acados":
            solver_class = AcadosSolver
        else:
            raise ValueError(f"Unknown solver {args.solver}")
        actor = MPCActor(solver_class=solver_class)
    else:
        actor = RLActor()

    # Initialize environment and actor
    env = AvantGoalEnv(num_envs=1, dt=0.1, time_limit_s=30, device='cpu', eval=True)
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
        
        obs, reward, done, _ = env.step(action.reshape(1, -1).astype(np.float32))

        if done:
            actor.reset()

        frame = env.render(mode='rgb_array', horizon=horizon, indices=[0])
        pygame.surfarray.blit_array(screen, frame.transpose(1, 0, 2))
        pygame.display.flip()

        # Ensure each frame lasts long enough to simulate 0.25x speed
        elapsed = pygame.time.get_ticks() - start_ticks
        if elapsed < frame_duration_ms:
            pygame.time.delay(frame_duration_ms - elapsed)