import torch
from stable_baselines3 import HerReplayBuffer
from stable_baselines3.common.vec_env import VecMonitor, VecVideoRecorder
from stable_baselines3.common.callbacks import CheckpointCallback

from sparse_to_dense_reward.avant_goal_env import AvantGoalEnv
from sparse_to_dense_reward.lac.lac import LAC
from sparse_to_dense_reward.utils import CustomCombinedExtractor

env = AvantGoalEnv(1000, 0.1, 30, "cuda:0")
env = VecMonitor(env)
env = VecVideoRecorder(env, "./RL_outputs/video", record_video_trigger=lambda x: x % 1000 == 0, video_length=500)

checkpoint_callback = CheckpointCallback(
  save_freq=1000,
  save_path="./logs/",
  name_prefix="rl_model",
  save_replay_buffer=True,
  save_vecnormalize=False,
)

n_sampled_goal = 5
model = LAC(
    "MultiInputPolicy",
    env,
    replay_buffer_class=HerReplayBuffer,
    replay_buffer_kwargs=dict(
        n_sampled_goal=n_sampled_goal,
        goal_selection_strategy="future"
    ),
    verbose=1,
    buffer_size=int(8e5),
    learning_starts=1000*400,
    learning_rate=5e-3,
    gradient_steps=3,
    train_freq=1,
    gamma=1.0,
    batch_size=int(2e5),
    policy_kwargs=dict(
        net_arch=dict(
            pi=[27, 54, 108, 54, 27],
            qf=[27, 54, 108, 54, 27],
        ),
        activation_fn=torch.nn.Softplus,
        n_critics=2,
        share_features_extractor=True,
        features_extractor_class=CustomCombinedExtractor
    ),
    tensorboard_log="./RL_outputs/debug",
)

# model.actor.load_state_dict(torch.load("avant_actor"))

custom_objects = { 'learning_rate': 1e-4, 'gradient_steps': 5, 'learning_starts': 1000*400, 'gamma': 1.0}
model = LAC.load("/home/aleksi/thesis_ws/src/rl_model_7000000_steps.zip", env=env, custom_objects=custom_objects)
model.load_replay_buffer("/home/aleksi/thesis_ws/src/rl_model_replay_buffer_7000000_steps.pkl")

model.learn(50e6, callback=checkpoint_callback)
model.save("lac")
model.save_replay_buffer("lac_buffer")
