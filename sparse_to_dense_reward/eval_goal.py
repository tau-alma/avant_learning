
from avant_goal_env import AvantGoalEnv
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecNormalize, VecMonitor, VecVideoRecorder

env = AvantGoalEnv(1000, 45, "cuda:0")
env = VecMonitor(env)
env = VecVideoRecorder(env, "./src/sparse_to_dense_reward/video", record_video_trigger=lambda x: x % 1000 == 0, video_length=200)

model = SAC.load("sac_vant", env=env)
evaluate_policy(model, env, n_eval_episodes=2000, render=True)