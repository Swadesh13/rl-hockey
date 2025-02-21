#!/usr/bin/env python3
import json
import os
import random
from glob import glob

import numpy as np
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.logger import configure
from utils.load import LoadTD7Agents

from henv.env import BasicOpponent, HockeyEnv_SB3, HockeyEnv_SB3_RND
from ppo.load_ppo_models import (
    eval_against_all_models,
    load_all_ppo_agents,
    load_all_sac_agents,
    load_ppo_agent,
)
from ppo.ppo import PPO_HockeyAgent
from utils.parsing import (
    get_config_from_args,
    get_eval_env,
    parse_args,
    print_args,
    print_config,
)


def train_against_many(
    opponent_dict,
    training_mode="random",
    num_iters=20,
    timesteps_per_iter=50_000,
    load=None,
    eval_episodes=50,
):
    # Parse standard configuration and any additional command-line args
    args = parse_args()
    cfg = get_config_from_args(args, cfgnode=True)
    cfg.logging.tensorboard = "./ppo/logs/allNL/"

    if not args.quiet:
        print_config(cfg)
        print_args(args)

    # Create environment (vectorized if specified in config)
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

    # Initialize the PPO agent and load the starting model (e.g. your ppo_pp model)
    agent = PPO_HockeyAgent(env, config=cfg)

    print()
    print(f"\tVerbose: {agent.model.verbose}")
    print(f"\tTensorboard log: {agent.model.tensorboard_log}")
    print(f"\tSave directory: {agent.save_dir}")
    print()

    if load == "yes":
        agent.load()
        agent.model.verbose = cfg.logging.verbose
        agent.model.tensorboard_log = agent.save_dir
        with open(os.path.join(agent.save_dir, "load.txt"), "w") as f:
            f.write("Model loaded successfully")
    elif load == "no" or load is None:
        with open(os.path.join(agent.save_dir, "NOT_load.txt"), "w") as f:
            f.write("Model not loaded")
    else:
        agent.load(load)
        agent.model.verbose = cfg.logging.verbose
        agent.model.tensorboard_log = agent.save_dir
        with open(os.path.join(agent.save_dir, "load.txt"), "w") as f:
            f.write("Model loaded successfully")

    print()
    print(f"\tVerbose: {agent.model.verbose}")
    print(f"\tTensorboard log: {agent.model.tensorboard_log}")
    print(f"\tSave directory: {agent.save_dir}")
    print()

    # Save the opponent dictionary to a file
    opponents_save_path = os.path.join(agent.save_dir, "opponents.json")
    with open(opponents_save_path, "w") as f:
        json.dump({k: str(v) for k, v in opponent_dict.items()}, f, indent=4)
    print(f"Saved opponents dictionary at {opponents_save_path}")

    eval_against_all_models(
        agent,
        opponent_dict,
        get_eval_env(),
        agent_name=args.config,
        num_episodes=eval_episodes,
        save_path=os.path.join(agent.save_dir, "eval_before.png"),
    )

    # Option 1: Stable (alternating) training: cycle through all opponents each iteration.
    if training_mode.lower() == "stable":
        print("\n--- Running stable alternating training ---\n")
        for cycle in range(num_iters):
            print(f"\n=== Cycle {cycle + 1} / {num_iters} ===")
            for opp_name, opp_model in opponent_dict.items():
                print(f"\n--- Training against {opp_name} ---")
                agent.set_opponent(opp_model, opponent_name=opp_name)
                agent.train(
                    total_timesteps=timesteps_per_iter,
                    log_interval=cfg.training.log_interval,
                )
                save_path = os.path.join(
                    agent.save_dir,
                    f"model_after_{opp_name}_cycle_{cycle + 1}",
                )
                agent.model.save(save_path)
                print(f"Saved model after training against {opp_name} at {save_path}")

    # Option 2: Random sampling training: each round picks a random opponent.
    elif training_mode.lower() == "random":
        print(
            f"\n--- Running random sampling training ({timesteps_per_iter} steps each) ---\n"
        )
        for rnd in range(num_iters):
            opp_name, opp_model = random.choice(list(opponent_dict.items()))
            print(
                f"\n=== Round {rnd + 1} / {num_iters}: Training against {opp_name} ==="
            )
            agent.set_opponent(opp_model, opponent_name=opp_name)
            agent.train(
                total_timesteps=timesteps_per_iter,
                log_interval=cfg.training.log_interval,
            )
            save_path = os.path.join(
                agent.save_dir, "chkpts", f"round_{rnd + 1}_{opp_name}"
            )
            agent.model.save(save_path)
            print(
                f"Saved model after round {rnd + 1} with opponent {opp_name} at {save_path}"
            )
            if rnd in [200,400,600,800,1000,1200,1400,1600,1800]:
                eval_against_all_models(
                    agent,
                    opponent_dict,
                    get_eval_env(),
                    agent_name=args.config,
                    num_episodes=eval_episodes,
                    save_path=os.path.join(agent.save_dir, f"eval_round_{rnd}.png"),
                )

    else:
        print(f"Unknown training_mode: {training_mode}. Choose 'stable' or 'random'.")

    eval_against_all_models(
        agent,
        opponent_dict,
        get_eval_env(),
        agent_name=args.config,
        num_episodes=eval_episodes,
        save_path=os.path.join(agent.save_dir, "eval_after.png"),
    )


if __name__ == "__main__":
    opponent_dict = {
        "basic_strong": BasicOpponent(weak=False),
        "basic_weak": BasicOpponent(weak=True),
        # "ppo_vanilla": load_ppo_agent("models/ppo/ppo_vanilla.yaml"),
    }
    opponent_dict.update(load_all_sac_agents())
    opponent_dict.update(load_all_ppo_agents())
    opponent_dict.update(LoadTD7Agents())

    train_against_many(
        opponent_dict,
        training_mode="random",
        num_iters=4000,
        timesteps_per_iter=50_000,
        load="no",
        eval_episodes=50
    )
