import json
import os
import random
from glob import glob

import numpy as np
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.logger import configure

from henv.env import BasicOpponent, HockeyEnv_SB3, HockeyEnv_SB3_RND
from ppo.load_ppo_models import (
    eval_against_all_models,
    load_all_ppo_agents,
    load_all_sac_agents,
    load_ppo_agent,
)
from ppo.ppo import PPO_HockeyAgent
from utils.load import LoadTD7Agents
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
# ALL_MODELS.update(LoadTD7Agents())
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
    """
    Train a PPO agent against multiple opponents in a specified training mode.

    Args:
        opponent_dict (dict): Dictionary of opponents to train against.
        training_mode (str): Training mode ('random' or 'stable').
        num_iters (int): Number of training iterations.
        timesteps_per_iter (int): Timesteps per iteration.
        load (str or None): Whether to load a pretrained model ('yes', 'no', or path).
        eval_episodes (int): Number of evaluation episodes.
        tensorboard_log (str or None): Path for tensorboard logs.
    """
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

    # Option 1: Stable training: cycle through all opponents each iteration.
    if training_mode.lower() == "stable":
        print("\n--- Running stable alternating training ---\n")
        for cycle in range(num_iters):
            print(f"\n=== Cycle {cycle + 1} / {num_iters} ===")
            for opp_name, opp_model in opponent_dict.items():
                print(
                    f"\n--- Training against {opp_name} for {timesteps_per_iter} steps ---"
                )
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
                opponent_dict,
                get_eval_env(),
                agent_name=args.config,
                num_episodes=eval_episodes,
                save_path=os.path.join(agent.save_dir, f"eval_round_{cycle+1}.png"),
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
        opponent_dict,
        get_eval_env(),
        agent_name=args.config,
        num_episodes=eval_episodes,
        save_path=os.path.join(agent.save_dir, "eval_after.png"),
    )


if __name__ == "__main__":

    # Define the opponents to train against
    opponent_dict = {}

    # Comment out the ones you don't want to train against

    # BASIC OPPONENTS
    opponent_dict["basic_strong"] = ALL_MODELS["basic_strong"]
    opponent_dict["basic_weak"] = ALL_MODELS["basic_weak"]

    # SAC OPPONENTS
    opponent_dict["sac_vanilla"] = ALL_MODELS["sac_vanilla"]
    opponent_dict["sac_pink"] = ALL_MODELS["sac_pink"]
    opponent_dict["sac_brown"] = ALL_MODELS["sac_brown"]
    opponent_dict["sac_reward"] = ALL_MODELS["sac_reward"]
    opponent_dict["sac_all_1"] = ALL_MODELS["sac_all_1"]

    # TD7 OPPONENTS (if you dont ahve GPU change the device to cpu in the td7 configs)
    # opponent_dict["td7_plain"] = ALL_MODELS["td7_plain"]
    # opponent_dict["td7_puck_proximity"] = ALL_MODELS["td7_puck_proximity"]
    # opponent_dict["td7_offensive_pressure"] = ALL_MODELS["td7_offensive_pressure"]
    # opponent_dict["td7_offensive_pressure_puck_proximity"] = ALL_MODELS[
    #     "td7_offensive_pressure_puck_proximity"
    # ]
    # opponent_dict["td7_all"] = ALL_MODELS["td7_all"]
    # opponent_dict["td7_offensive_pressure_self"] = ALL_MODELS[
    #     "td7_offensive_pressure_self"
    # ]
    # opponent_dict["td7_all_big"] = ALL_MODELS["td7_all_big"]
    # opponent_dict["td7_all_new"] = ALL_MODELS["td7_all_new"]
    # opponent_dict["td7_all_offfensive"] = ALL_MODELS["td7_all_offfensive"]
    # opponent_dict["td7_crash"] = ALL_MODELS["td7_crash"]

    # PPO OPPONENTS
    opponent_dict["ppo_vanilla"] = ALL_MODELS["ppo_vanilla"]
    opponent_dict["ppo_gaussian_noise"] = ALL_MODELS["ppo_gaussian_noise"]
    opponent_dict["ppo_offensive_pressure"] = ALL_MODELS["ppo_offensive_pressure"]
    opponent_dict["ppo_pp+op"] = ALL_MODELS["ppo_pp+op"]
    opponent_dict["ppo_puck_proximity"] = ALL_MODELS["ppo_puck_proximity"]
    opponent_dict["ppo_rnd_e1_i0.01"] = ALL_MODELS["ppo_rnd_e1_i0.01"]

    train_against_many(
        opponent_dict,
        training_mode="stable",
        num_iters=40,
        timesteps_per_iter=250_000,
        load="yes",
        eval_episodes=50,
        tensorboard_log="./ppo/logs/many/",
    )
