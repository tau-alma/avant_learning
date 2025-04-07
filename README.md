# Avant_learning
This repository contains the codes needed to train an [Adaptive Lyapunov-based Actor-Critic](https://openreview.net/pdf?id=rOCWUmMBSnH) agent to solve a wheel loader pose reaching task.

## Gym environment
The motion model and the rewards of the simulated environment are documented in [the corresponding paper](https://arxiv.org/pdf/2409.15717).

![image](https://github.com/user-attachments/assets/b9799416-033a-44d0-a128-8585ce966739)

The RL environment differs from a standard [gymnasium](https://gymnasium.farama.org/) environment in two ways:
1. It implements a goal-conditioned GoalEnv interface to support [Stable Baselines 3 implementation of Hindsight Experience Replay (HER)](https://stable-baselines.readthedocs.io/en/master/modules/her.html)
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
| 1 | scaled linear acceleration ($$\dot v_f$$, -1 to 1|

## ALAC RL agent

## Setup and execution
On a CUDA enabled PC, the dependencies for training can be installed with:
```bash
pip install -r requirements.txt
```

To train the RL agent, run the training script:
```bash
python teach_loader.py
```
this will commence a curriculum learning run, where the loader is first trained to reach a goal position, then a goal position + heading, then to terminate with zero center joint angle, and so on.

The resulting RL critic can be evaluated within the Actor-Critic MPC framework by running:
```bash
python test_loader_MPC.py
```
which will start a graphical user interface, through which the user can assign goal poses by left click + drag, and instansiate obstacles by right click + drag:

https://github.com/user-attachments/assets/05df6e7a-c63d-46de-8d9c-70530bd1aea6


