import os
import sys
from glob import glob

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.logger import configure

from henv.env import HockeyEnv_SB3
from henv.hockey_agent import HockeyAgent
from utils.evaluate import eval_agent
from utils.parsing import *
# from datetime import datetime
import random
import time


class PPO_HockeyAgent(HockeyAgent):
    def __init__(self, env, config):
        """
        Initializes the PPO agent for the Hockey environment.

        Parameters:
        - env: The environment instance.
        - config: Configuration node (CfgNode).
        """
        super().__init__(env, config)
        tnsr_dir = self.config.logging.tensorboard
        time.sleep(np.random.uniform(0, 15))
        self.save_dir = f"{tnsr_dir}PPO_{len(glob(f'{tnsr_dir}PPO_*'))}"
        # current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        # self.save_dir = f"{tnsr_dir}PPO_{current_time}"
        print("Saving files at:", self.save_dir)
        os.makedirs(self.save_dir, exist_ok=True)

        self.model = PPO(
            "MlpPolicy",
            self.env,
            verbose=self.config.logging.verbose,
            tensorboard_log=self.save_dir,
            # policy_kwargs=self.config.model.hyperparameters.get("policy_kwargs", {}),
            **self.config.model.hyperparameters,
        )

        # make sure it saves the tensorboard logs in the correct directory
        new_logger = configure(self.save_dir, ["tensorboard"])
        self.model.set_logger(new_logger)

        # save the configuration
        save_config(os.path.join(self.save_dir, "config.yaml"), self.config)

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

    from henv.env import HockeyEnv_SB3

    env = HockeyEnv_SB3.make_vec_env(n_envs=cfg.environment.n_envs)

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
        )
