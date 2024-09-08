import sys
import numpy as np
import json
import matplotlib.pyplot as plt
from PyQt6 import QtWidgets
from tqdm import tqdm
from multiprocessing import Queue, Process, Event
from argparse import ArgumentParser
from sparse_to_dense_reward.avant_goal_env import AvantGoalEnv
from sparse_to_dense_reward.qtplot import QtPlot
from mpc_solvers.acados_solver import AcadosSolver
from eval_helpers import MPCActor, TrajectoryPlanner, make_eval_plots
from sparse_to_dense_reward.utils import AvantRenderer

MACHINE_RADIUS = 0.8

def launch_qt_window(frame_size, data_queue, exit_event):
    app = QtWidgets.QApplication(sys.argv)
    plot = QtPlot(frame_size, data_queue, exit_event)
    plot.show()
    app.exec()  # Blocks until window is closed
    if not exit_event.is_set():
        exit_event.set()

def rl_env_test_loop(scenario_file, device, rebuild):
    scenario_given = scenario_file is not None
    if scenario_given:
        with open(scenario_file, "r") as f:
            scenario = json.load(f)
            initial_pose = np.asarray([scenario["pose"]], dtype=np.float32)
            goal_pose = np.asarray([scenario["goal_pose"]], dtype=np.float32)
            obstacles = np.asarray(scenario["obstacles"], dtype=np.float32)
    else:
        obstacles = []

    actor = MPCActor(solver_class=AcadosSolver, device=device, obstacles=obstacles)
    trajectory_planner = TrajectoryPlanner(rebuild=rebuild, obstacles=obstacles) # To compare against

    # Initialize environment and actor
    env = AvantGoalEnv(num_envs=1, dt=0.01, time_limit_s=30, device='cpu', eval=True, eval_input_delay=False)

    # Initialize the plotter
    data_queue = Queue()
    exit_event = Event()
    import atexit
    def stop():
        exit_event.set()
    atexit.register(stop)
    plot_process = Process(target=launch_qt_window, args=(env.renderer.RENDER_RESOLUTION, data_queue, exit_event))
    plot_process.start()

    history = []

    if scenario_given:
        obs = env.reset(initial_pose, goal_pose)   
        comparision, comparision_time = np.empty(0), 0 #trajectory_planner.plan(obs)
    else:     
        obs = env.reset()
        comparision, comparision_time = [], 0

    while True:
        if exit_event.is_set():
            break

        action, horizon, solve_time, lyapunov_value = actor.act(obs)
        horizon = horizon[:, :4]
        action /= env.dynamics.control_scalers.cpu().numpy()

        if scenario_given:
            history.append(np.r_[horizon[0], lyapunov_value])
        
        # Substep euler physics for more accuracy:
        for i in range(10):
            obs, reward, done, _ = env.step(action.reshape(1, -1).astype(np.float32))
            if done:
                break

        if done:
            if scenario_given:
                exit_event.set()
                break
            else:
                actor.reset()

        frame = env.render(indices=[0], mode='rgb_array', horizon=horizon[1:], obstacles=obstacles, comparision=comparision)
        data_queue.put((solve_time, lyapunov_value, frame))

    if scenario_given:
        history = np.asarray(history)
        make_eval_plots(history, comparision, comparision_time, scenario_file)


def stability_test_loop(device):
    renderer = AvantRenderer(6)
    actor = MPCActor(solver_class=AcadosSolver, device=device)

    data_queue = Queue()
    exit_event = Event()
    import atexit
    def stop():
        exit_event.set()
    atexit.register(stop)
    plot_process = Process(target=launch_qt_window, args=(1280, data_queue, exit_event))
    plot_process.start()

    goals = []
    distances = [4.5, 9]
    angles = [0, 1/4*np.pi, 2/4*np.pi, 3/4*np.pi, np.pi, 5/4*np.pi, 6/4*np.pi, 7/4*np.pi]
    for dist in distances:
        for offset in angles:
            for goal_heading in angles:
                goals.append([dist * np.cos(offset), dist * np.sin(offset), goal_heading])
    goals = np.asarray(goals)
    print(f"N GOALS: {len(goals)}")

    def is_done(x, goal):
        achieved_hdg = x[2]
        desired_hdg = goal[2]
        hdg_error = np.arctan2(np.sin(achieved_hdg - desired_hdg), np.cos(achieved_hdg - desired_hdg))
        e = np.r_[x[:2] - goal[:2], hdg_error]
        return np.linalg.norm(e) < 0.1

    lyapunov_rate_values = []
    time_to_goals = []
    succesful_scenarios = np.repeat(False, len(goals))
    g_i = 0
    for goal in tqdm(goals):
        current_rate_values = []
        x = np.zeros(6)
        last_lyapunov_value = None
        success = False
        for i in range(int(30 / actor.problem.h)):
            if exit_event.is_set():
                return

            if is_done(x, goal):
                success = True
                break

            _, horizon, solve_time, lyapunov_value = actor.solve(x, goal)
            x = horizon[1]
            if last_lyapunov_value is not None:
                current_rate_values.append((lyapunov_value - last_lyapunov_value) / actor.problem.h)
            last_lyapunov_value = lyapunov_value

            frame = renderer.render(state=x, goal=goal, horizon=horizon[1:, :4], obstacles=[], comparision=[])
            data_queue.put((solve_time, lyapunov_value, frame))

        if not success:
            print("FAILED with goal", goal)
        else:
            lyapunov_rate_values.extend(current_rate_values)
            succesful_scenarios[g_i] = True
        time_to_goals.append(i*actor.problem.h)
        g_i += 1

    plt.figure(figsize=(10, 10))
    plt.grid(True)
    plt.gca().set_axisbelow(True)
    plt.quiver(0, 0, 1, 0, angles='xy', scale_units='xy', scale=0.75, label="Initial pose", color="g")
    i1 = np.argwhere(succesful_scenarios)
    i2 = np.argwhere(~succesful_scenarios)
    plt.quiver(goals[i1, 0], goals[i1, 1], np.cos(goals[i1, 2]), np.sin(goals[i1, 2]), angles='xy', scale_units='xy', scale=1, label="Goal poses", color="black")
    plt.quiver(goals[i2, 0], goals[i2, 1], np.cos(goals[i2, 2]), np.sin(goals[i2, 2]), angles='xy', scale_units='xy', scale=1, label="Goal poses", color="red")
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.xticks([-9, -6, -3, 0, 3, 6, 9])
    plt.yticks([-9, -6, -3, 0, 3, 6, 9])
    plt.xlabel('X', fontsize=20)
    plt.ylabel('Y', fontsize=20)
    # plt.title('')
    plt.legend(prop={'size': 20})
    plt.tight_layout()
    plt.gca().tick_params(labelsize=20)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig("empirical_scenarios.svg")

    print(len(lyapunov_rate_values))
    lyapunov_rate_values = np.asarray(lyapunov_rate_values)
    print(np.amin(lyapunov_rate_values), np.amax(lyapunov_rate_values))
    lyapunov_rate_values = np.clip(lyapunov_rate_values, -50, 50)

    fig, ax = plt.subplots(figsize=(10, 10))
    plt.grid(True)
    ax.hist(lyapunov_rate_values, bins=100)
    plt.tight_layout()
    plt.savefig(f"lyapunov_rates_hist.svg", bbox_inches='tight')
    plt.close()

def main():
    parser = ArgumentParser()
    parser.add_argument("--scenario-file", type=str)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--stability", action="store_true")
    args = parser.parse_args()

    device = "cuda:0" if args.cuda else "cpu"

    if not args.stability:
        rl_env_test_loop(args.scenario_file, device, args.rebuild)
    else: 
        stability_test_loop(device)


if __name__ == "__main__":
    main()