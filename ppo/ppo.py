import os

# from datetime import datetime
import random
import sys
import time
from glob import glob

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.noise import (
    NormalActionNoise,
    OrnsteinUhlenbeckActionNoise,
)

from henv.env import HockeyEnv_SB3
from henv.hockey_agent import HockeyAgent
from utils.evaluate import eval_agent
from utils.noise import ColoredActionNoise, ColoredNoiseProcess
from utils.parsing import *


class PPO_HockeyAgent(HockeyAgent):
    def __init__(self, env, config):
        """
        Initializes the PPO agent for the Hockey environment.

        Parameters:
        - env: The environment instance.
        - config: Configuration node (CfgNode).
        """
        super().__init__(env, config)
        self.use_rnd = config.rnd.enabled

        self._create_logging_dir()

        # Initialize noise
        self.noise = None
        noise_type = config.model.get("noise", None)  # Get noise type as a string
        if noise_type:
            self.noise = self._initialize_noise(noise_type, env)
            print(f"Initialized noise: {self.noise}")

        # Initialize policy kwargs
        policy_kwargs = self.config.model.hyperparameters.get("policy_kwargs", {})
        policy_kwargs = cfg_node_to_dict(policy_kwargs)
        policy_kwargs["activation_fn"] = get_activation_fn_from_str(
            policy_kwargs["activation_fn"]
        )
        # if self.use_rnd:
        #     policy_kwargs["net_arch"] = [
        #         dict(vf=[256, 256], pi=[256, 256])
        #     ]  # Two heads

        # Remove `policy_kwargs` from hyperparameters to avoid duplication
        hyperparameters = self.config.model.hyperparameters.copy()
        hyperparameters.pop("policy_kwargs", None)

        self.model = PPO(
            "MlpPolicy",
            self.env,
            verbose=self.config.logging.verbose,
            tensorboard_log=self.save_dir,
            policy_kwargs=policy_kwargs,
            **hyperparameters,
        )

        # make sure it saves the tensorboard logs in the correct directory
        new_logger = configure(self.save_dir, ["tensorboard"])
        self.model.set_logger(new_logger)

        # save the configuration
        save_config(os.path.join(self.save_dir, "config.yaml"), self.config)

    def train(
        self,
        total_timesteps=None,
        log_interval=None,
        progress_bar=False,
        callbacks=None,
    ):
        """
        Overrides the train method to include optional action noise.
        """
        if self.noise:
            print(f"Applying noise: {self.noise}")
            self.model.policy.noise = self.noise  # Add noise to policy during training

        super().train(total_timesteps, log_interval, progress_bar, callbacks)

    def load(self, load_path=None):
        """
        Loads the PPO model.
        """
        path = load_path or self.model_path
        if os.path.exists(f"{path}.zip"):
            self.model = PPO.load(path, env=self.env)
            print(f"Model loaded from {path}")
        else:
            print(f"No model found at {path}. Starting with a new model.")

    def _initialize_noise(self, noise_type, env):
        """
        Initializes the appropriate noise type based on the string input.

        Parameters:
        - noise_type (str): Type of noise (e.g., 'pink', 'brown', 'white', 'gaussian', 'ornstein').
        - env: The environment instance to get action dimensions.

        Returns:
        - ActionNoise object or None if no valid noise type is provided.
        """
        action_dim = env.action_space.shape[0]  # Default action dimension
        seq_len = 250  # Fixed sequence length
        sigma = 0.3

        if noise_type.lower() == "pink":
            return ColoredActionNoise(
                beta=1.0, sigma=sigma, seq_len=seq_len, action_dim=action_dim
            )
        elif noise_type.lower() == "brown":
            return ColoredActionNoise(
                beta=2.0, sigma=sigma, seq_len=seq_len, action_dim=action_dim
            )
        elif noise_type.lower() == "white":
            return ColoredActionNoise(
                beta=0.0, sigma=sigma, seq_len=seq_len, action_dim=action_dim
            )
        elif noise_type.lower() == "gaussian":
            return NormalActionNoise(
                mean=np.zeros(action_dim), sigma=sigma * np.ones(action_dim)
            )
        elif noise_type.lower() == "ornstein":
            return OrnsteinUhlenbeckActionNoise(
                mean=np.zeros(action_dim), sigma=sigma, theta=0.15
            )
        else:
            raise ValueError(
                f"Unsupported noise type: {noise_type}. Choose from 'pink', 'brown', 'white', 'gaussian', 'ornstein'."
            )

    def _create_logging_dir(self):
        """
        Creates a logging directory for the PPO agent.
        """
        tnsr_dir = self.config.logging.tensorboard
        time.sleep(np.random.uniform(0, 15))
        self.save_dir = f"{tnsr_dir}PPO_{len(glob(f'{tnsr_dir}PPO_*'))}"
        print("Saving files at:", self.save_dir)
        os.makedirs(self.save_dir, exist_ok=True)


if __name__ == "__main__":
    args = parse_args()
    cfg = get_config_from_args(args, cfgnode=True)

    if not args.quiet:
        print_config(cfg)
        print_args(args)

    from henv.env import HockeyEnv_SB3, HockeyEnv_SB3_RND

    if cfg.rnd.enabled:
        env = HockeyEnv_SB3_RND.make_vec_env_rnd(
            n_envs=cfg.environment.n_envs,
            config=cfg,
            weak_opponent=False,
            additional_rewards=cfg.environment.additional_rewards,
            reward_multiplier=cfg.environment.reward_multiplier,
        )
    else:
        env = HockeyEnv_SB3.make_vec_env(
            n_envs=cfg.environment.n_envs,
            weak_opponent=False,
            additional_rewards=cfg.environment.additional_rewards,
            reward_multiplier=cfg.environment.reward_multiplier,
        )

    agent = PPO_HockeyAgent(env, config=cfg)

    if args.train:
        from stable_baselines3.common.monitor import Monitor

        callback_list = []
        # Define callbacks
        checkpoint_callback = CheckpointCallback(
            save_freq=int(cfg.training.save_model_every / cfg.environment.n_envs),
            save_path=f"{agent.save_dir}/chkpts",
            name_prefix="ppo_model",
        )

        print("creating eval env")
        eval_env = Monitor(HockeyEnv_SB3())
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=f"{agent.save_dir}/best_model/",
            log_path=f"{agent.save_dir}/ppo_eval_logs/",
            eval_freq=1000,
            deterministic=True,
            render=False,
        )
        callback_list.append(checkpoint_callback)
        callback_list.append(eval_callback)

        agent.train(
            total_timesteps=cfg.training.total_timesteps,
            log_interval=cfg.training.log_interval,
            progress_bar=False,
            callbacks=callback_list,
        )
        # agent.save()

    if args.eval:
        agent.load()
        agent.evaluate(
            num_episodes=args.eval_episodes,
            render_mode="human" if not args.no_render else "rgb_array",
            opponent_right=None,
            modes=["NORMAL", "TRAIN_SHOOTING", "TRAIN_DEFENSE"],
        )
