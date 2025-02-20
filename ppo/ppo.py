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
from utils.noise import ColoredActionNoise
from utils.parsing import *


class PPO_HockeyAgent(HockeyAgent):
    def __init__(self, env, config, eval=False):
        """
        Initializes the PPO agent for the Hockey environment.

        Parameters:
        - env: The environment instance.
        - config: Configuration node (CfgNode).
        """
        super().__init__(env, config)
        self.use_rnd = config.rnd.enabled

        if not eval:
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

        # Remove `policy_kwargs` from hyperparameters to avoid duplication
        hyperparameters = self.config.model.hyperparameters.copy()
        hyperparameters.pop("policy_kwargs", None)

        if eval:
            self.model = PPO(
                "MlpPolicy",
                self.env,
                verbose=self.config.logging.verbose,
                policy_kwargs=policy_kwargs,
                **hyperparameters,
            )
        else:
            self.model = PPO(
                "MlpPolicy",
                self.env,
                verbose=self.config.logging.verbose,
                tensorboard_log=self.save_dir,
                policy_kwargs=policy_kwargs,
                **hyperparameters,
            )

        if not eval:
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

    def set_opponent(self, opponent, opponent_name=None):
        """
        Set a new opponent for the environment.
        The opponent must implement a `predict()` method.
        """
        if hasattr(self.env, "envs"):
            for single_env in self.env.envs:
                single_env.opponent = opponent
            print("Opponent updated in all vectorized environments to", opponent_name)
        else:
            self.env.opponent = opponent
            print("Opponent updated in the single environment to", opponent_name)


if __name__ == "__main__":
    args = parse_args()
    cfg = get_config_from_args(args, cfgnode=True)

    if not args.quiet:
        print_config(cfg)
        print_args(args)

    if args.train:
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

        agent.load()  # TODO make a parser for this

        # --- CODE TO SET A DIFFERENT OPPONENT ---
        # For example, load a different PPO agent as the opponent.
        from ppo.load_ppo_models import load_ppo_agent

        opponent_config_path = "models/ppo/ppo_vanilla.yaml"
        opponent_agent = load_ppo_agent(opponent_config_path)
        agent.set_opponent(opponent_agent, opponent_name=opponent_config_path)
        # --------------------------------------------------

        print("Training agent...")
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
        print("Evaluating agent...")
        import hockey.hockey_env as h_env
        from stable_baselines3.common.monitor import Monitor

        from henv.env import BasicOpponent
        from ppo.load_ppo_models import (
            eval_against_all_models,
            load_all_ppo_agents,
            load_all_sac_agents,
            load_ppo_agent,
        )

        WEAK = False
        eval_env = h_env.HockeyEnv_BasicOpponent(weak_opponent=WEAK)

        agent = PPO_HockeyAgent(eval_env, config=cfg, eval=True)

        agent.load()
        print(f"==> Loaded main agent PPO from config: {args.config}")

        # === ONLY ONE OPPONENT ===

        # # opponent = BasicOpponent(weak=WEAK)
        # opponent_cfg = "models/ppo/ppo_vanilla.yaml"
        # opponent = load_ppo_agent(opponent_cfg)
        # print("==> Loaded opponent:", opponent_cfg)

        # agent.evaluate(
        #     num_episodes=args.eval_episodes,
        #     render_mode="human" if not args.no_render else "rgb_array",
        #     opponent_right=opponent,
        #     modes=["NORMAL"],  # "TRAIN_SHOOTING", "TRAIN_DEFENSE"
        #     env=eval_env,
        #     verbose=2,
        # )

        # === ALL OPPONENTS ===
        models = load_all_ppo_agents()
        models.update(load_all_sac_agents())
        models["basic_weak"] = BasicOpponent(weak=True)
        models["basic_strong"] = BasicOpponent(weak=False)

        eval_against_all_models(
            agent,
            models,
            eval_env,
            agent_name=args.config,
            num_episodes=args.eval_episodes,
        )

    if not args.train and not args.eval:
        print("No action specified. add --train or --eval to run the agent.")
