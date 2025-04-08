# Avant_learning
This repository contains the codes needed to train an [Adaptive Lyapunov-based Actor-Critic](https://openreview.net/pdf?id=rOCWUmMBSnH) agent to solve a wheel loader pose reaching task.


## Gym environment
The motion model and the rewards of the [simulated environment](loader_navigation_rl/loader_goal_env.py) are documented in [the corresponding paper](https://arxiv.org/pdf/2409.15717).

![image](https://github.com/user-attachments/assets/b9799416-033a-44d0-a128-8585ce966739)

The RL environment differs from a standard [gymnasium](https://gymnasium.farama.org/) environment in two ways:
1. It implements a goal-conditioned GoalEnv interface to support [Stable Baselines 3 implementation of Hindsight Experience Replay (HER)](https://stable-baselines3.readthedocs.io/en/master/modules/her.html)
2. The vectorization is done by leveraging CUDA computations within the environment, rather than relying on CPU multiprocessing

Consequently, the observation space of the environment is a dictionary with vector-valued fields:

|  Dictionary key  |  Shape  | Description |
|------|------|------|
| obs["desired_goal"] | (num_envs, 7) | The current goal |
| obs["achieved_goal"] | (num_envs, 7) | The current state |

where each state is a vector with fields:
|  State index  |  Description |
|------|------|
| 0 | $$x_f$$|
| 1 | $$y_f$$|
| 2 | $$\sin(\theta_f)$$|
| 3 | $$\cos (\theta_f)$$|
| 4 | $$\beta$$|
| 5 | $$\dot \beta$$|
| 6 | $$v_f$$|

The step function of the environment expects to receive a vector of actions: (num_envs, 2), where each action consists of:
|  Action index  |  Description |
|------|------|
| 0 | scaled center joint acceleration ($$\ddot \beta$$), -1 to 1|
| 1 | scaled linear acceleration ($$\dot v_f$$), -1 to 1|


## ALAC RL agent
The [ALAC algorithm](loader_navigation_rl/alac) is implemented as a subclass of Stable Baselines 3 [off-policy algorithm](https://stable-baselines3.readthedocs.io/en/master/_modules/stable_baselines3/common/off_policy_algorithm.html).

The actor and critic both expect as input a feature vector of shape (batch_size, 7), which has been extracted from the dictionary observation as follows:
```bash
achieved = self.extractors["achieved_goal"](observations["achieved_goal"])
desired = self.extractors["desired_goal"](observations["desired_goal"])
obs = self.extractors["observation"](observations["observation"])
pos_residual = desired[:, :2] - achieved[:, :2]
achieved_hdg_data = achieved[:, 2:4]
desired_hdg_data = desired[:, 2:4]
sin_achieved, cos_achieved = achieved_hdg_data[:, 0], achieved_hdg_data[:, 1]
sin_desired, cos_desired = desired_hdg_data[:, 0], desired_hdg_data[:, 1]
# Compute the position error in the front body unit frame:
rotation_matrix = torch.stack([
    cos_achieved, sin_achieved, 
    -sin_achieved, cos_achieved
], dim=1).reshape(-1, 2, 2)
local_pos_residual = torch.bmm(rotation_matrix, pos_residual.unsqueeze(-1)).squeeze(-1)
# Recover the heading error from sin/cos:
hdg_error = torch.atan2(
    sin_desired * cos_achieved - cos_desired * sin_achieved,
    cos_desired * cos_achieved + sin_desired * sin_achieved
)
encoded_tensor_list = [
    local_pos_residual,                # longitudinal and lateral error
    torch.sin(hdg_error.unsqueeze(1)), # sin(heading error)
    torch.cos(hdg_error.unsqueeze(1)), # cos(heading error)
    obs                                # beta, dot_beta, lin_vel
]
encoded_tensors = torch.cat(encoded_tensor_list, dim=1)
```
i.e. the position error has been transformed into the coordinate frame of the loader front body unit, and the heading error has been encoded using sine and cosine values.

When using the critic as a MPC cost function, two things need to be considered:
1. By default, the [feature extractor](loader_navigation_rl/utils) is not saved as part of the critic, therefore the correct input vector needs to be manually constructed in the MPC formulation (as shown in the [example](test_loader_mpc.py))
2. The critic output needs to be "manually" squared (this is done automatically within the [SymbolicMPCProblem](https://github.com/tau-alma/ACMPC-solvers/blob/master/mpc_problem.py) class)


## Setup and execution
On a CUDA enabled PC, the dependencies for training can be installed with:
```bash
pip install -r requirements.txt
```

To train the RL agent, run the training script:
```bash
python teach_loader.py
```
this will commence a curriculum learning run, where the loader is first trained to reach a goal position, then a goal position + heading, then to terminate with zero center joint angle, and so on. The training progress is logged in tensorboard format to "./RL_outputs/", and videos of the training rollouts are saved in "./RL_outputs/videos".

The resulting RL critic can be evaluated within the Actor-Critic MPC framework by running:
```bash
python test_loader_MPC.py
```
which will start a graphical user interface, through which the user can assign goal poses by left click + drag, and instansiate obstacles by right click + drag:

https://github.com/user-attachments/assets/05df6e7a-c63d-46de-8d9c-70530bd1aea6


