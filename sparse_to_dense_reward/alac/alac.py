from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import get_parameters_by_name, polyak_update

from sparse_to_dense_reward.utils import compute_gradient_penalty
from sparse_to_dense_reward.alac.policies import Actor, ALACPolicy, MultiInputPolicy
from sparse_to_dense_reward.alac.utils import SquaredContinuousCritic

SelfALAC = TypeVar("SelfALAC", bound="ALAC")


class ALAC(OffPolicyAlgorithm):
    """
    Adaptive Lyapunov-based Actor Critic

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: learning rate for adam optimizer,
        the same learning rate will be used for all networks (Q-Values, Actor and Value function)
        it can be a function of the current progress remaining (from 1 to 0)
    :param buffer_size: size of the replay buffer
    :param learning_starts: how many steps of the model to collect transitions for before learning starts
    :param batch_size: Minibatch size for each gradient update
    :param tau: the soft update coefficient ("Polyak update", between 0 and 1)
    :param gamma: the discount factor
    :param train_freq: Update the model every ``train_freq`` steps. Alternatively pass a tuple of frequency and unit
        like ``(5, "step")`` or ``(2, "episode")``.
    :param gradient_steps: How many gradient steps to do after each rollout (see ``train_freq``)
        Set to ``-1`` means to do as many gradient steps as steps done in the environment
        during the rollout.
    :param action_noise: the action noise type (None by default), this can help
        for hard exploration problem. Cf common.noise for the different action noise type.
    :param replay_buffer_class: Replay buffer class to use (for instance ``HerReplayBuffer``).
        If ``None``, it will be automatically selected.
    :param replay_buffer_kwargs: Keyword arguments to pass to the replay buffer on creation.
    :param optimize_memory_usage: Enable a memory efficient variant of the replay buffer
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
    :param ent_coef: Entropy regularization coefficient. (Equivalent to
        inverse of reward scale in the original SAC paper.)  Controlling exploration/exploitation trade-off.
        Set it to 'auto' to learn it automatically (and 'auto_0.1' for using 0.1 as initial value)
    :param target_update_interval: update the target network every ``target_network_update_freq``
        gradient steps.
    :param target_entropy: target entropy when learning ``ent_coef`` (``ent_coef = 'auto'``)
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param use_sde_at_warmup: Whether to use gSDE instead of uniform sampling
        during the warm up phase (before learning starts)
    :param stats_window_size: Window size for the rollout logging, specifying the number of episodes to average
        the reported success rate, mean episode length, and mean reward over
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    :param alpha3: The Lyapunov decrease coefficient
    """

    policy_aliases: ClassVar[Dict[str, Type[BasePolicy]]] = {
        "MultiInputPolicy": MultiInputPolicy,
    }

    policy: ALACPolicy
    actor: Actor
    critic: SquaredContinuousCritic
    critic_target: SquaredContinuousCritic

    def __init__(
        self,
        policy: Union[str, Type[ALACPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = 1,
        gradient_steps: int = 1,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class: Optional[Type[ReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        target_update_interval: int = 1,
        target_entropy: Union[str, float] = "auto",
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        use_sde_at_warmup: bool = False,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        regularize: bool = False,
    ):
        super().__init__(
            policy,
            env,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            policy_kwargs=policy_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            seed=seed,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            use_sde_at_warmup=use_sde_at_warmup,
            optimize_memory_usage=optimize_memory_usage,
            supported_action_spaces=(spaces.Box,),
            support_multi_env=True,
        )

        self.target_update_interval = target_update_interval
        self.target_entropy = target_entropy
        self.regularize = regularize
    
        # Optimizers for the lyapunov loss lagrange variables:
        self.log_beta = th.tensor([1.0], device=self.device).requires_grad_(True)
        self.beta_optimizer: Optional[th.optim.Adam] = None
        self.log_llambda = th.tensor([1e-2], device=self.device).requires_grad_(True)
        self.llambda_optimizer: Optional[th.optim.Adam] = None

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()
        self._create_aliases()
        # Running mean and running var
        self.batch_norm_stats = get_parameters_by_name(self.critic, ["running_"])
        self.batch_norm_stats_target = get_parameters_by_name(self.critic_target, ["running_"])

        # Target entropy is used when learning the entropy coefficient
        if self.target_entropy == "auto":
            # automatically set target entropy if needed
            self.target_entropy = float(-np.prod(self.env.action_space.shape).astype(np.float32))  # type: ignore
            print("----------------------------------")
            print(self.target_entropy)
        else:
            # Force conversion
            # this will also throw an error for unexpected string
            self.target_entropy = float(self.target_entropy)

        self.llambda_optimizer = th.optim.Adam([self.log_llambda], lr=self.lr_schedule(1))
        self.beta_optimizer = th.optim.Adam([self.log_beta], lr=self.lr_schedule(1))

    def _create_aliases(self) -> None:
        self.actor = self.policy.actor
        self.critic = self.policy.critic
        self.critic_target = self.policy.critic_target

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizers learning rate
        optimizers = [self.actor.optimizer, self.critic.optimizer]

        # Update learning rate according to lr schedule
        self._update_learning_rate(optimizers)

        actor_losses, critic_losses = [], []
        llambda_losses, beta_losses = [], []

        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]

            # We need to sample because `log_std` may have changed between two gradient steps
            if self.use_sde:
                self.actor.reset_noise()

            # Action by the current actor for the sampled state
            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            # Select action according to policy
            next_actions_pi, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
            # Compute the next L values

            L_values = th.cat(self.critic_target(replay_data.observations, replay_data.actions), dim=1)
            L_values, _ = th.max(L_values, dim=1, keepdim=True)
            next_L_values = th.cat(self.critic_target(replay_data.next_observations, next_actions_pi), dim=1)
            next_L_values, _ = th.max(next_L_values, dim=1, keepdim=True)
            
            # td error
            target_L_values = -replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_L_values.detach()
            # Get current L-values estimates for each critic network
            # using action from the replay buffer
            current_L_values = self.critic(replay_data.observations, replay_data.actions)

            # Compute critic loss
            critic_loss = 0.5 * sum(F.mse_loss(current_L, target_L_values) for current_L in current_L_values)
            if self.regularize:
                critic_loss += compute_gradient_penalty(self.critic, replay_data.observations, replay_data.actions, lambda_gp=1e-5)
            assert isinstance(critic_loss, th.Tensor)  # for type checker
            critic_losses.append(critic_loss.item())   # type: ignore[union-attr]

            # Optimize the critic
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # Ensure positive lagrange multipliers:
            beta = th.exp(self.log_beta)
            llambda = th.exp(self.log_llambda)

            k = 1 - llambda.detach().item()
            lam = min(llambda.detach().item(), self.gamma)
            delta_L = next_L_values - L_values.detach() + k*(L_values.detach() - lam*next_L_values)

            # Compute actor loss, i.e. solve the inner min problem of the lagrangian:
            actor_loss = th.mean(
                beta.detach()*log_prob
                + llambda.detach()*delta_L
            )
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()
            actor_losses.append(actor_loss.item())

            beta_loss = -beta * th.mean(self.target_entropy + log_prob.detach())
            self.beta_optimizer.zero_grad()
            beta_loss.backward()
            self.beta_optimizer.step()
            beta_losses.append(beta_loss.item())
            
            llambda_loss = -llambda * th.mean(delta_L.detach())
            self.llambda_optimizer.zero_grad()
            llambda_loss.backward()
            self.llambda_optimizer.step()
            llambda_losses.append(llambda_loss.item())

            # Early on during the training there will be large violations of the lyapunov constraint due to a bad policy,
            # therefore it is wise to clamp the associated lagrange multiplier from above, so it doesn't dominate the gradients:
            with th.no_grad():
                self.log_beta.clamp_(max=np.log(1))
                self.log_llambda.clamp_(max=np.log(1))

            # Update target networks
            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                # Copy running stats, see GH issue #996
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        self.logger.record("train/beta_loss", np.mean(beta_losses))
        self.logger.record("train/lambda_loss", np.mean(llambda_losses))
        self.logger.record("train/beta", th.exp(self.log_beta).cpu().detach().item())
        self.logger.record("train/lambda", th.exp(self.log_llambda).cpu().detach().item())
        self.logger.record("train/k", k)
        self.logger.record("train/lam", lam)

    def learn(
        self: SelfALAC,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "ALAC",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfALAC:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )

    def _excluded_save_params(self) -> List[str]:
        return super()._excluded_save_params() + ["actor", "critic", "critic_target"]  # noqa: RUF005

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "actor.optimizer", "critic.optimizer", "beta_optimizer", "llambda_optimizer"]
        saved_pytorch_variables = ["log_beta", "log_llambda"]
        return state_dicts, saved_pytorch_variables
