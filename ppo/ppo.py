import os
import sys

import numpy as np
import torch
from stable_baselines3 import PPO

from utils.evaluate import eval_agent
from utils.parsing import *


class HockeyPPOAgent:
    def __init__(self, env, config=None):
        """
        Initializes the PPO agent for the Hockey environment.

        Parameters:
        - env: The environment instance.
        - config: Configuration node (CfgNode).
        """
        self.env = env
        self.config = config or get_default_ppo_config(cfgnode=True)
        if not self.config.training.model_path:
            raise ValueError(
                "Model path for training is not specified in the configuration."
            )
        self.model_path = self.config.training.model_path

        hyperparameters = self.config.model.hyperparameters
        self.model = PPO(
            "MlpPolicy",
            self.env,
            **hyperparameters,
            verbose=self.config.logging.verbose,
        )

    def train(self, total_timesteps=None):
        """
        Trains the PPO model.

        Parameters:
        - total_timesteps: Total timesteps for training.
        """
        tt = total_timesteps or self.config.training.total_timesteps

        print("Starting training...")
        self.model.learn(total_timesteps=tt)
        print("Training complete.")

    def evaluate(self, num_episodes=10, render_mode="human", opponent_right=None):
        """
        Evaluates the trained PPO model in the environment.

        Parameters:
        - num_episodes: Number of episodes to evaluate (overrides config if provided).
        - render_mode: Mode for rendering the environment (overrides config if provided).
        - opponent_right: Optional opponent for the evaluation.

        Returns:
        - mean_reward: Mean reward over all episodes.
        - std_reward: Standard deviation of rewards over all episodes.
        """

        return eval_agent(
            self.model,
            opponent_right=opponent_right,
            num_episodes=num_episodes,
            render_mode=render_mode,
        )

    def save(self, save_path=None):
        """
        Saves the trained PPO model.
        """
        path = save_path or self.model_path
        self.model.save(path)
        print(f"Model saved at {path}")

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


if __name__ == "__main__":
    args = parse_args()
    cfg = get_config_from_args(args, cfgnode=True)

    if not args.quiet:
        print_config(cfg)
        print_args(args)

    from env import HockeyEnv_SB3

    env = HockeyEnv_SB3.make_vec_env(n_envs=cfg.environment.n_envs)

    agent = HockeyPPOAgent(env, config=cfg)

    if args.train:
        agent.train(total_timesteps=cfg.training.total_timesteps)
        agent.save()

    if args.eval:
        agent.load()
        agent.evaluate(num_episodes=args.eval_episodes)
