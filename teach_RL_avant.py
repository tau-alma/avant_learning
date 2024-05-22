import torch
from sparse_to_dense_reward.avant_goal_env import AvantGoalEnv
from stable_baselines3 import PPO, SAC, HerReplayBuffer
from stable_baselines3.sac.policies import MultiInputPolicy
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecNormalize, VecMonitor, VecVideoRecorder
from sparse_to_dense_reward.utils import CustomCombinedExtractor
from ogm_encoder.vae import VAE

encoder = VAE(256, 64).cuda()
encoder.load_state_dict(torch.load("/home/aleksi/thesis_ws/src/outputs/vae_20.ckpt"))
encoder.eval()
for param in encoder.parameters():
    param.requires_grad = False

env = AvantGoalEnv(1000, 0.2, 30, "cuda:0", num_obstacles=5, encoder=encoder)
env = VecMonitor(env)
# env = VecNormalize(env)
env = VecVideoRecorder(env, "./src/sparse_to_dense_reward/video", record_video_trigger=lambda x: x % 1000 == 0, video_length=500)

n_sampled_goal = 3

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
    learning_starts=1000*160,
    learning_rate=2.5e-3,
    gradient_steps=10,
    train_freq=1,
    gamma=0.999,
    batch_size=int(1e5),
    policy_kwargs=dict(
        net_arch=dict(
            pi=[512]*5, 
            qf=[512]*5
        ),
        n_critics=2,
        share_features_extractor=True,
        features_extractor_class=CustomCombinedExtractor
    ),
    tensorboard_log="./src/sparse_to_dense_reward/debug"
)

model.learn(5e6)
model.save("sac_vant")
