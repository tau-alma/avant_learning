import casadi as cs
import numpy as np
import config
import torch
import sys
import time
from queue import Empty
from PyQt6 import QtWidgets, QtGui, QtCore
from multiprocessing import Process, Queue, Event
from mpc_solvers.mpc_problem import SymbolicMPCProblem, SymbolicMPCSolver
from mpc_solvers.acados_sqp_solver import AcadosSolver
from loader_rendering.renderer import LoaderRenderer


MACHINE_RADIUS = LoaderRenderer.MACHINE_RADIUS


class MPCActor:
    def __init__(self, solver_class: SymbolicMPCSolver, num_obstacles=0, mpc_n=10, mpc_d=0.0):
        super().__init__()
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
            -fake_inf, -fake_inf, -fake_inf, -config.loader_max_beta, -config.loader_max_dot_beta,
            config.loader_min_v
        ])
        ubx_vec = np.array([
            fake_inf, fake_inf, fake_inf, config.loader_max_beta, config.loader_max_dot_beta, 
            config.loader_max_v
        ])
        ocp_x_slacks = {3: 1000, 4: 1000, 5: 1000}

        # Controls
        dot_dot_beta = cs.MX.sym("dot_dot_beta")
        a_f = cs.MX.sym("a_f")
        ocp_u = cs.vertcat(dot_dot_beta, a_f)
        lbu_vec = np.array([
            -config.loader_max_dot_dot_beta, -config.loader_max_a
        ])
        ubu_vec = np.array([
            config.loader_max_dot_dot_beta, config.loader_max_a,
        ])

        # Params
        x_goal = cs.MX.sym("x_goal")
        y_goal = cs.MX.sym("y_goal")
        theta_goal = cs.MX.sym("y_goal")
        ocp_p = cs.vertcat(x_goal, y_goal, theta_goal)

        obstacles = []
        for i in range(num_obstacles):
            o_x = cs.MX.sym(f"ox_{i}")
            o_y = cs.MX.sym(f"oy_{i}")
            o_r = cs.MX.sym(f"or_{i}")
            ocp_p = cs.vertcat(ocp_p, o_x, o_y, o_r)
            obstacles.append([o_x, o_y, o_r])
        
        # Continuous dynamics:
        omega_f = -(
            (config.loader_lr * dot_beta_ref + v_f_ref * cs.sin(beta)) 
            / (config.loader_lf * cs.cos(beta) + config.loader_lr)
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

        # Initialize parametric obstacles:
        lbg_vec = None
        ubg_vec = None
        g_fun = None
        g_slacks = {}
        t_g_fun = None
        if num_obstacles > 0:
            lbg_vec = np.zeros(num_obstacles)
            ubg_vec = np.repeat(fake_inf, num_obstacles)
            g_expr = cs.vertcat(
                *[ 
                    (x_f - o[0])**2 + (y_f - o[1])**2 - (o[2] + MACHINE_RADIUS)**2   
                    for o in obstacles          
                ]
            )
            g_fun = cs.Function('g', [ocp_x, ocp_u, ocp_p], [g_expr])
            t_g_fun = cs.Function('g', [ocp_x, ocp_p], [g_expr])
            g_slacks = {i: 100 for i in range(num_obstacles)}

        self.problem = SymbolicMPCProblem(
            N=mpc_n,
            h=0.2,
            input_delay=mpc_d,
            ocp_x=ocp_x,
            lbx_vec=lbx_vec,
            ubx_vec=ubx_vec,
            ocp_x_slacks=ocp_x_slacks,
            ocp_u=ocp_u,
            lbu_vec=lbu_vec,
            ubu_vec=ubu_vec,
            ocp_p=ocp_p,
            dynamics_fun=f,
            lbg_vec=lbg_vec,
            ubg_vec=ubg_vec,
            g_fun=g_fun,
            ocp_g_slacks=g_slacks,
            terminal_lbg_vec=lbg_vec,
            terminal_ubg_vec=ubg_vec,
            terminal_g_fun=t_g_fun,
            ocp_t_g_slacks=g_slacks
        )
 
        # Terminal cost with Q network:
        x_err = x_goal - x_f
        y_err = y_goal - y_f
        critic_stage_state = cs.vertcat(
            cs.cos(theta_f) * x_err + cs.sin(theta_f) * y_err, -cs.sin(theta_f) * x_err + cs.cos(theta_f) * y_err,    
            cs.sin(theta_goal - theta_f), cs.cos(theta_goal - theta_f), 
            beta, dot_beta_ref, v_f_ref, 
            dot_dot_beta / config.loader_max_dot_dot_beta, a_f / config.loader_max_a
        )       
        critic_terminal_state = cs.vertcat(
            cs.cos(theta_f) * x_err + cs.sin(theta_f) * y_err, -cs.sin(theta_f) * x_err + cs.cos(theta_f) * y_err,   
            cs.sin(theta_goal - theta_f), cs.cos(theta_goal - theta_f), 
            beta, dot_beta_ref, v_f_ref, 
            0, 0
        )
        critic_model = torch.load("loader_critic").eval()
        self.problem.add_stage_neural_cost(model=critic_model, model_state=critic_stage_state)
        self.problem.add_terminal_neural_cost(model=critic_model, model_state=critic_terminal_state)
        self.solver = solver_class(self.problem)

    def reset(self):
        pass            

    def solve(self, x0, goal, obstacles) -> np.ndarray:      
        params = [np.r_[goal, obstacles.flatten()] for _ in range(self.problem.N + 1)]
        
        t1 = time.time_ns()
        sol_x, sol_u, sol = self.solver.solve(x0, params)

        terminal_state = self.problem.ocp_x_to_terminal_state_fun(sol_x[-1], params[-1])
        lyapunov_value = self.problem.terminal_cost_fun(terminal_state, params[-1]).full()[0]

        t2 = time.time_ns()
        solve_time = ((t2-t1)/1e6)

        controls = sol_u[0, :2]
        
        return controls, sol_x, solve_time, lyapunov_value
    

class ImageWindow(QtWidgets.QLabel):
    def __init__(self, res, dist, goal_queue, obstacle_queue, frame_queue):
        super().__init__()
        self.res = res
        self.dist = dist
        self.max_n_obstacles = max_n_obstacles
        self.goal_queue = goal_queue
        self.frame_queue = frame_queue
        self.obstacle_queue = obstacle_queue
        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop | QtCore.Qt.AlignmentFlag.AlignLeft)
        self.setMouseTracking(True)
        self.l_click_pos = None
        self.r_click_pos = None
        self.setMinimumSize(res, res)

        self.timer = QtCore.QTimer()
        self.timer.setInterval(30)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start()

    def update_frame(self):
        try:
            frame = self.frame_queue.get()
            if frame.dtype != np.uint8:
                frame = (255 * np.clip(frame, 0, 1)).astype(np.uint8)
            h, w, _ = frame.shape
            image = QtGui.QImage(frame.data, w, h, 3 * w, QtGui.QImage.Format.Format_RGB888)
            pixmap = QtGui.QPixmap.fromImage(image)
            self.setPixmap(pixmap)
        except Empty:
            pass

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self.l_click_pos = event.position()
            event.accept()
        elif event.button() == QtCore.Qt.MouseButton.RightButton:
            print("Right click")
            self.r_click_pos = event.position()
            event.accept()

    def mouseReleaseEvent(self, event):
        def px_to_world(px_x, px_y):
            scale = (4 * self.dist) / self.res
            x_rel = px_x - self.res / 2
            y_rel = px_y - self.res / 2

            world_x = -x_rel * scale
            world_y = y_rel * scale
            return world_x, world_y
        
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            if self.l_click_pos is not None:
                release_pos = event.position()
                x_world, y_world = px_to_world(release_pos.x(), release_pos.y())
                x_start, y_start = px_to_world(self.l_click_pos.x(), self.l_click_pos.y())
                dx_world = x_world - x_start
                dy_world = y_world - y_start
                theta = np.arctan2(dy_world, dx_world)
                goal = np.array([x_world, y_world, theta], dtype=np.float32)
                self.goal_queue.put(goal)
                self.l_click_pos = None
                event.accept()
        elif event.button() == QtCore.Qt.MouseButton.RightButton:
            if self.r_click_pos is not None:
                release_pos = event.position()
                x_world, y_world = px_to_world(release_pos.x(), release_pos.y())
                x_start, y_start = px_to_world(self.r_click_pos.x(), self.r_click_pos.y())
                radius = np.linalg.norm(np.array([x_world - x_start, y_world - y_start]))
                obstacle = np.array([x_start, y_start, radius], dtype=np.float32)
                self.obstacle_queue.put(obstacle)
                self.r_click_pos = None
                event.accept()

    
def main_loop(res, dist, max_n_obstacles, goal_queue, obstacle_queue, frame_queue, event):
    lr = LoaderRenderer(res, dist, False)
    actor = MPCActor(AcadosSolver, mpc_n=20, num_obstacles=max_n_obstacles)

    x = np.zeros(6)
    goal = np.zeros(3)

    n_obstacles = 0
    obstacles = np.tile(np.array([1e3, 1e3, 0]), max_n_obstacles).reshape(max_n_obstacles, 3)
    while True:
        if event.is_set():
            break

        try:
            goal = goal_queue.get(block=False)
        except Empty:
            pass

        try:
            obstacle = obstacle_queue.get(block=False)
            obstacles[n_obstacles] = obstacle
            n_obstacles = (n_obstacles + 1) % max_n_obstacles
        except Empty:
            pass

        _, horizon, solve_time, lyapunov_value = actor.solve(x, goal, obstacles)
        x = horizon[1]
        frame = lr.render_frame(state=x, goal=goal, horizon=horizon[1:, :4], obstacles=obstacles, comparision=[])
        frame_queue.put(frame)

if __name__ == "__main__":
    res = 1024
    dist = 5
    max_n_obstacles = 3

    goal_queue = Queue()
    obstacle_queue = Queue()
    frame_queue = Queue()
    kill_event = Event()
    process = Process(target=main_loop, 
                      args=(res, dist, max_n_obstacles, goal_queue, obstacle_queue, frame_queue, kill_event))
    process.start()

    app = QtWidgets.QApplication([])
    window = ImageWindow(res, dist, goal_queue, obstacle_queue, frame_queue)
    window.show()
    window.setWindowTitle("Loader MPC")

    try:
        app.exec()
    except KeyboardInterrupt:
        print("Received Ctrl+C. Shutting down.")
    finally:
        kill_event.set()
        if process.is_alive():
            process.terminate()
            process.join()
        sys.exit(0)