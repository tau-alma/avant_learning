from avant_goal_env import AvantGoalEnv
from stable_baselines3 import PPO, HerReplayBuffer, SAC
from stable_baselines3.sac.policies import MultiInputPolicy
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecNormalize, VecMonitor, VecVideoRecorder

env = AvantGoalEnv(1000, 45, "cuda:0")
env = VecMonitor(env)
env = VecVideoRecorder(env, "./src/sparse_to_dense_reward/video", record_video_trigger=lambda x: x % 1000 == 0, video_length=1000)

# Create 2 artificial transitions per real transition
n_sampled_goal = 2

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
    learning_starts=1000*304,
    learning_rate=1e-3,
    gradient_steps=10,
    gamma=0.999,
    batch_size=102400,
    policy_kwargs=dict(net_arch=dict(pi=[96, 64, 32, 16], qf=[96, 64, 32, 16], n_critics=1)),
    tensorboard_log="./src/sparse_to_dense_reward/debug"
)
model.learn(50e6)
model.save("sac_vant")
