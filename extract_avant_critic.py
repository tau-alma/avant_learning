import torch
from sparse_to_dense_reward.avant_goal_env import AvantGoalEnv
from stable_baselines3 import SAC

env = AvantGoalEnv(1000, 0.2, 30, "cuda:0")
model = SAC.load("sac_vant", env=env)
torch.save(model.critic.q_networks[0], "avant_critic")
