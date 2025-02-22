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

ALL_MODELS = {
    "basic_strong": BasicOpponent(weak=False),
    "basic_weak": BasicOpponent(weak=True),
}
ALL_MODELS.update(load_all_sac_agents())
ALL_MODELS.update(LoadTD7Agents())
ALL_MODELS.update(load_all_ppo_agents())



def train_against_many(
    opponent_dict,
    training_mode="random",
    num_iters=20,
    timesteps_per_iter=50_000,
    load=None,
    eval_episodes=50,
    tensorboard_log=None,
):
    # Parse standard configuration and any additional command-line args
    args = parse_args()
    cfg = get_config_from_args(args, cfgnode=True)
    if tensorboard_log is not None:
        cfg.logging.tensorboard = tensorboard_log

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
        ALL_MODELS,
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
                    "chkpts",
                    f"model_after_{opp_name}_cycle_{cycle + 1}",
                )
                agent.model.save(save_path)
                print(f"Saved model after training against {opp_name} at {save_path}")
                
            eval_against_all_models(
                agent,
                ALL_MODELS,
                get_eval_env(),
                agent_name=args.config,
                num_episodes=eval_episodes,
                save_path=os.path.join(agent.save_dir, f"eval_round_{cycle}.png"),
            )

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
            if rnd in list(range(200, num_iters, 200)):
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
        ALL_MODELS,
        get_eval_env(),
        agent_name=args.config,
        num_episodes=eval_episodes,
        save_path=os.path.join(agent.save_dir, "eval_after.png"),
    )


if __name__ == "__main__":
    opponent_dict = {
        # "basic_strong": BasicOpponent(weak=False),
        # "basic_weak": BasicOpponent(weak=True),
    }
    opponent_dict.update(load_all_sac_agents())
    opponent_dict.pop("sac_brown", None)
    # opponent_dict.pop("sac_pink", None)
    opponent_dict.pop("sac_reward", None)
    # opponent_dict.pop("sac_vanilla", None)

    opponent_dict.update(LoadTD7Agents())
    opponent_dict.pop("td7_all", None)
    opponent_dict.pop("td7_offensive_pressure", None)
    opponent_dict.pop("td7_offensive_pressure_puck_proximity", None)
    # opponent_dict.pop("td7_offensive_pressure_self", None)
    # opponent_dict.pop("td7_plain", None)
    opponent_dict.pop("td7_puck_proximity", None)

    # opponent_dict.update(load_all_ppo_agents())
    # opponent_dict.pop("ppo_gaussian_noise", None)
    # opponent_dict.pop("ppo_offensive_pressure", None)
    # opponent_dict.pop("ppo_pp+op", None)
    # opponent_dict.pop("ppo_puck_proximity", None)
    # opponent_dict.pop("ppo_rnd_e1_i0.01", None)
    # opponent_dict.pop("ppo_vanilla", None)
    ALL_MODELS = opponent_dict

    train_against_many(
        opponent_dict,
        training_mode="stable",
        num_iters=1000,
        timesteps_per_iter=250_000,
        load="yes",
        eval_episodes=50,
        tensorboard_log="./ppo/logs/op_sac_td7/",
    )