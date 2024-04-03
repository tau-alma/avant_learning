from avant_goal_env import AvantGoalEnv
from stable_baselines3 import PPO, HerReplayBuffer, SAC
from stable_baselines3.sac.policies import MlpPolicy
from stable_baselines3.common.vec_env import VecNormalize, VecMonitor, VecVideoRecorder
import gymnasium as gym
import numpy as np


def learning_rate_schedule(initial_lr=1e-3, scale_factor=1e-1, scale_every_steps=30e6):
    def lr_schedule(current_step):
        scaling_events = np.floor(current_step / scale_every_steps)
        current_lr = initial_lr * (scale_factor ** scaling_events)
        return current_lr
    return lr_schedule


env = AvantGoalEnv(1000, 30, "cuda:0")
env = VecMonitor(env)
env = VecNormalize(env)
env = VecVideoRecorder(env, "./src/sparse_to_dense_reward/video", record_video_trigger=lambda x: x % 1000 == 0, video_length=1200)
print(env.observation_space)

# Create 2 artificial transitions per real transition
n_sampled_goal = 2

model = SAC(
    "MultiInputPolicy",
    env,
    replay_buffer_class=HerReplayBuffer,
    replay_buffer_kwargs=dict(
        n_sampled_goal=n_sampled_goal,
        goal_selection_strategy="future",
        copy_info_dict=True
    ),
    verbose=1,
    buffer_size=int(2e6),
    learning_starts=1000*304,
    learning_rate=1e-3,
    gradient_steps=10,
    gamma=0.999,
    batch_size=102400,
    policy_kwargs=dict(net_arch=dict(pi=[64, 64, 64, 64, 64], qf=[64, 64, 64, 64, 64])),
    tensorboard_log="./src/sparse_to_dense_reward/debug"
)
# model = SAC.load("sac_vant", env=env)
model.learn(20e6)
model.save("sac_vant")

