from sparse_to_dense_reward.avant_goal_env import AvantGoalEnv
from stable_baselines3 import PPO, HerReplayBuffer, SAC
from stable_baselines3.sac.policies import MultiInputPolicy
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecNormalize, VecMonitor, VecVideoRecorder
from sparse_to_dense_reward.utils import CustomCombinedExtractor

env = AvantGoalEnv(2000, 0.2, 30, "cuda:0")
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
    learning_starts=2000*304,
    learning_rate=2.5e-3,
    gradient_steps=10,
    train_freq=1,
    gamma=0.999,
    batch_size=int(1e5),
    policy_kwargs=dict(
        net_arch=dict(
            pi=[128]*5, 
            qf=[128]*5
        ),
        n_critics=2,
        share_features_extractor=True,
        features_extractor_class=CustomCombinedExtractor   
    ),
    tensorboard_log="./src/sparse_to_dense_reward/debug"
)
model.set_parameters("sac_vant_1")
model.learn(10e6)
model.save("sac_vant")
