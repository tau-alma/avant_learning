import os
import torch
import numpy as np
from stable_baselines3.common.buffers import DictReplayBuffer
from stable_baselines3.common.vec_env import VecMonitor, VecVideoRecorder
from stable_baselines3.common.callbacks import CheckpointCallback
from loader_navigation_rl.loader_goal_env import LoaderGoalEnv
from loader_navigation_rl.alac.alac import ALAC
from loader_navigation_rl.utils import CustomCombinedExtractor
from datetime import datetime

now = datetime.now().strftime("%m_%d__%H_%M")
os.makedirs(f"RL_outputs/{now}", exist_ok=True)

env = LoaderGoalEnv(1000, 0.2, 25, "cuda:0")
env = VecMonitor(env)
env = VecVideoRecorder(env, "./RL_outputs/video", record_video_trigger=lambda x: (x > 1000) and (x % 50*151 == 0), video_length=5*151)

checkpoint_callback = CheckpointCallback(
  save_freq=1000,
  save_path=f"RL_outputs/{now}",
  name_prefix="rl_model",
  save_replay_buffer=True,
  save_vecnormalize=False,
)

model = ALAC(
    "MultiInputPolicy",
    env,
    replay_buffer_class=DictReplayBuffer,
    verbose=1,
    buffer_size=int(2e6),
    learning_starts= int(1e6),
    learning_rate=1e-3,
    gradient_steps=3,
    train_freq=1,
    gamma=0.999,
    batch_size=int(3e5),
    policy_kwargs=dict(
        net_arch=dict(
            pi=[96, 144, 96],
            qf=[48, 96, 48],
        ),
        activation_fn=torch.nn.Softplus,
        share_features_extractor=True,
        features_extractor_class=CustomCombinedExtractor
    ),
    tensorboard_log="./RL_outputs/debug",
    target_entropy=-1.5,
    tau=0.01,
    lambda_gp=0.0
)

try:
    sd = torch.load("avant_critic").state_dict()
    sd2 = {}
    for k, v in sd.items():
        sd2["qf0."+k] = v
        sd2["qf1."+k] = v
        model.policy.critic.load_state_dict(sd2, strict=True)
        model.critic.load_state_dict(sd2, strict=True)
        model.policy.critic_target.load_state_dict(sd2, strict=True)
        model.critic_target.load_state_dict(sd2, strict=True)
except:
    print("Couldn't load critic weights")

try:
    model.policy.actor.load_state_dict(torch.load("avant_actor"))
    model.actor.load_state_dict(torch.load("avant_actor"))
except:
    print("Couldn't load actor weights")
    
# Critic learns a bit faster than actor:
for g in model.policy.critic.optimizer.param_groups:
    g['lr'] = 3e-2
for g in model.critic.optimizer.param_groups:
    g['lr'] = 3e-2
for g in model.policy.actor.optimizer.param_groups:
    g['lr'] = 1.5e-3
for g in model.actor.optimizer.param_groups:
    g['lr'] = 1.5e-3

pos_w = 1/0.1
hdg_w = 1/np.deg2rad(5)
beta_w = 1/np.deg2rad(5)
dot_beta_w = 1/np.deg2rad(25)
lin_vel_w = 1
pos_beta_w = 5
pos_dot_beta_w = 10

# It makes training easier if we introduce the error terms one by one as the total steps grow, i.e.:
# position error weight: 1     1     1     1     1     1  
# hdg error weight:      0     1     1     1     1     1
# beta error weight:     0     0     1     1     1     and so on...
#                        -----------------------------------------
# example total steps:   0    1M    2M    4M    8M    16M
weight_schedule = [
    ({"pos": pos_w, "hdg": 0, "beta": 0, "dot_beta": 0, "lin_vel": 0, "pos_beta": 0, "pos_dot_beta": 0}, int(5e6)),
    ({"pos": pos_w, "hdg": hdg_w, "beta": 0, "dot_beta": 0, "lin_vel": 0, "pos_beta": 0, "pos_dot_beta": 0}, int(5e6)),
    ({"pos": pos_w, "hdg": hdg_w, "beta": beta_w, "dot_beta": 0, "lin_vel": 0, "pos_beta": 0, "pos_dot_beta": 0}, int(5e6)),
    ({"pos": pos_w, "hdg": hdg_w, "beta": beta_w, "dot_beta": dot_beta_w, "lin_vel": 0, "pos_beta": 0, "pos_dot_beta": 0}, int(5e6)),
    ({"pos": pos_w, "hdg": hdg_w, "beta": beta_w, "dot_beta": dot_beta_w, "lin_vel": lin_vel_w, "pos_beta": 0, "pos_dot_beta": 0}, int(5e6)),
    ({"pos": pos_w, "hdg": hdg_w, "beta": beta_w, "dot_beta": dot_beta_w, "lin_vel": lin_vel_w, "pos_beta": pos_beta_w, "pos_dot_beta": 0}, int(5e6)),
    ({"pos": pos_w, "hdg": hdg_w, "beta": beta_w, "dot_beta": dot_beta_w, "lin_vel": lin_vel_w, "pos_beta": pos_beta_w, "pos_dot_beta": pos_dot_beta_w}, int(5e6)),
]

total_step = 0
for w, steps in weight_schedule:
    env.env.set_cost_weights(w)
    model.learn(steps, callback=checkpoint_callback)
    total_step += steps
    model.save(f"RL_outputs/{now}/lac_{int(steps/1e6)}.zip")
    model.save_replay_buffer(f"RL_outputs/{now}/lac_buffer_{int(steps/1e6)}.pkl")
    model.replay_buffer.reset() # Clear the outdated buffer with old rewards, and collect new samples in next iter