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
from mpc_solvers.acados_solver import AcadosSolver
from avant_modeling.gp import GPModel

class MPCActor:
    def __init__(self, solver_class, device):
        super().__init__()
        self.device = device
        self.history = []
        fake_inf = 1e7
        
        self.gp_dict = {}
        for name in ["delayed_u_steer", "delayed_u_gas"]:
            data_x = torch.load(f"sparse_to_dense_reward/{name}/{name}_gp_inputs.pth").to(device)
            data_y = torch.load(f"sparse_to_dense_reward/{name}/{name}_gp_targets.pth").to(device)
            gp = GPModel(data_x, data_y, train_epochs=0, device=device).to(device)
            gp.load_state_dict(torch.load(f"sparse_to_dense_reward/{name}/{name}_gp_model.pth"))
            self.gp_dict[name] = gp
        
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
        ocp_p = cs.vertcat(x_goal, y_goal, theta_goal)

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

        # accel_penalty = 1e-3*dot_dot_beta**2 + 1e-3*a_f**2
        # standstill_turning_penalty = (1 - cs.tanh(v_f_ref**2 / 0.05)) * (dot_beta_ref)**2
        l = cs.Function('l', [ocp_x, ocp_u, ocp_p], [0])

        self.problem = SymbolicMPCProblem(
            N=10,
            h=0.1,
            input_delay=0.0,
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
        critic_stage_state = cs.vertcat(
            x_goal - x_f,       y_goal - y_f,    
            cs.sin(theta_f),    cs.cos(theta_f), 
            cs.sin(theta_goal), cs.cos(theta_goal), 
            beta, dot_beta_ref, v_f_ref, 
            dot_dot_beta / config.avant_max_dot_dot_beta, a_f / config.avant_max_a
        )
        critic_terminal_state = cs.vertcat(
            x_goal - x_f,       y_goal - y_f,    
            cs.sin(theta_f),    cs.cos(theta_f), 
            cs.sin(theta_goal), cs.cos(theta_goal), 
            beta, dot_beta_ref, v_f_ref, 
            0, 0
        )
        critic_model = torch.load("avant_critic").eval()
        self.problem.add_stage_neural_cost(model=critic_model, model_state=critic_stage_state)
        self.problem.add_terminal_neural_cost(model=critic_model, model_state=critic_terminal_state)

        self.solver = solver_class(self.problem, rebuild=True)

    def reset(self):
        pass            

    def _compute_machine_controls(self, x: np.ndarray):
        beta = x[3]
        dot_beta = x[4]
        v_f = x[5]

        with torch.no_grad():
            gp_input = torch.tensor([beta, dot_beta, v_f], dtype=torch.float32).unsqueeze(0).to(self.device)
            steer = self.gp_dict["delayed_u_steer"](gp_input).mean.cpu().numpy()[0]
            gas = self.gp_dict["delayed_u_gas"](gp_input).mean.cpu().numpy()[0]

        return steer, gas

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
        t_s = self.problem.ocp_x_to_terminal_state_fun(obs, params[0])
        t_v = self.problem.terminal_cost_fun(t_s, params[0])
        controls = self._compute_machine_controls(sol_x[1, :])

        t2 = time.time_ns()
        self.history.append((t2-t1)/1e6)
        print(t_v, np.asarray(self.history).mean())

        return controls, sol_x[2:, :4]
    

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--solver", type=str)
    parser.add_argument("--scenario-file", type=str)
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()

    if args.solver == "casadi_collocation":
        solver_class = CasadiCollocationSolver
    elif args.solver == "acados":
        solver_class = AcadosSolver
    else:
        raise ValueError(f"Unknown solver {args.solver}")
    device = "cuda:0" if args.cuda else "cpu"
    actor = MPCActor(solver_class=solver_class, device=device)

    # Initialize environment and actor
    env = AvantGoalEnv(num_envs=1, dt=0.01, time_limit_s=30, device='cpu', eval=True)
    screen = pygame.display.set_mode([env.renderer.RENDER_RESOLUTION, env.renderer.RENDER_RESOLUTION])

    # Frame rate and corresponding real-time frame duration
    frame_rate = 30
    normal_frame_duration_ms = 1000 // frame_rate
    frame_duration_ms = normal_frame_duration_ms

    # Main simulation loop
    obs = env.reset()

    history = []
    while True:
        for _ in pygame.event.get():
            pass
        start_ticks = pygame.time.get_ticks()  # Get the start time of the current frame

        action, horizon = actor.act(obs)
        action /= env.dynamics.control_scalers.cpu().numpy()
        
        # Substep for more accuracy:
        for i in range(10):
            obs, reward, done, _ = env.step(action.reshape(1, -1).astype(np.float32))
            if done:
                break

        if done:
            actor.reset()

        frame = env.render(mode='rgb_array', horizon=horizon, indices=[0])
        pygame.surfarray.blit_array(screen, frame.transpose(1, 0, 2))
        pygame.display.flip()

        # Ensure each frame lasts long enough to simulate 0.25x speed
        elapsed = pygame.time.get_ticks() - start_ticks
        if elapsed < frame_duration_ms:
            pygame.time.delay(frame_duration_ms - elapsed)