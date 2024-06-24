import torch
from sparse_to_dense_reward.avant_goal_env import AvantGoalEnv
from stable_baselines3 import PPO, SAC, HerReplayBuffer
from stable_baselines3.sac.policies import MultiInputPolicy
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecNormalize, VecMonitor, VecVideoRecorder
from sparse_to_dense_reward.utils import CustomCombinedExtractor
from ogm_encoder.vae import VAE
from stable_baselines3.common.callbacks import CheckpointCallback


env = AvantGoalEnv(1000, 0.2, 30, "cuda:0", num_obstacles=3)
env = VecMonitor(env)
# env = VecNormalize(env)
env = VecVideoRecorder(env, "./RL_outputs/video", record_video_trigger=lambda x: x % 1000 == 0, video_length=400)

checkpoint_callback = CheckpointCallback(
  save_freq=1000,
  save_path="./logs/",
  name_prefix="rl_model",
  save_replay_buffer=True,
  save_vecnormalize=False,
)

n_sampled_goal = 5
model = SAC(
    "MultiInputPolicy",
    env,
    replay_buffer_class=HerReplayBuffer,
    replay_buffer_kwargs=dict(
        n_sampled_goal=n_sampled_goal,
        goal_selection_strategy="future"
    ),
    verbose=1,
    buffer_size=int(2e6),
    learning_starts=1000*200,
    learning_rate=1e-3,
    gradient_steps=10,
    train_freq=1,
    gamma=0.9999,
    batch_size=int(2e5),
    policy_kwargs=dict(
        net_arch=dict(
            pi=[18, 36, 72, 36, 18],
            qf=[18, 36, 72, 36, 18],
        ),
        activation_fn=torch.nn.Softplus,
        n_critics=2,
        share_features_extractor=True,
        features_extractor_class=CustomCombinedExtractor
    ),
    tensorboard_log="./RL_outputs/debug"
)
print(model.critic)
print(model.actor)

#custom_objects = { 'learning_rate': 5e-5 }
#model = SAC.load("/home/aleksi/thesis_ws/src/rl_model_5000000_steps.zip", env=env, custom_objects=custom_objects)
#model.load_replay_buffer("/home/aleksi/thesis_ws/src/rl_model_replay_buffer_5000000_steps.pkl")

model.learn(10e6, callback=checkpoint_callback)
model.save("sac")
model.save_replay_buffer("sac_buffer")
