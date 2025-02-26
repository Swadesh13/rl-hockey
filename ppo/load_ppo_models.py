import gc

import hockey.hockey_env as h_env
import matplotlib.pyplot as plt
import pandas as pd
import pygame

from ppo.ppo import PPO_HockeyAgent
from sac.sac import SAC_HockeyAgent
from utils.parsing import convert_to_cfgnode, get_eval_env, load_config


def load_ppo_agent(config_path: str):
    """
    Load a PPO agent from a saved model.

    Args:
        config_path (str): Path to the PPO configuration file.

    Returns:
        PPO_HockeyAgent: Loaded PPO agent.

    models/ppo/ppo_vanilla.yaml
    models/ppo/ppo_gaussian_noise.yaml
    models/ppo/ppo_offensive_pressure.yaml
    models/ppo/ppo_pp+op.yaml
    models/ppo/ppo_puck_proximity.yaml
    models/ppo/ppo_rnd_e1_i0.01.yaml
    """

    cfg = convert_to_cfgnode(load_config(config_path))

    eval_env = get_eval_env()
    agent = PPO_HockeyAgent(eval_env, config=cfg, eval=True)

    agent.load()
    return agent


def load_all_ppo_agents():
    """
    Returns:
        dict: Dictionary of loaded PPO agents. {agent_name: agent}
    """
    models = [
        "ppo_vanilla",
        "ppo_gaussian_noise",
        "ppo_offensive_pressure",
        "ppo_pp+op",
        "ppo_puck_proximity",
        "ppo_rnd_e1_i0.01",
    ]

    agents = {}
    for model in models:
        agents[model] = load_ppo_agent(f"models/ppo/{model}.yaml")

    return agents


def load_all_sac_agents():
    """
    Load all saved SAC agents.

    Returns:
        dict: Dictionary of loaded SAC agents.
    """
    cfg = convert_to_cfgnode(load_config("configs/sac_hockey.yaml"))

    eval_env = get_eval_env()
    models = {}

    sac_vanilla = SAC_HockeyAgent(eval_env, config=cfg)
    sac_vanilla.load("models/sac/sac_vanilla")
    models["sac_vanilla"] = sac_vanilla

    sac_pink = SAC_HockeyAgent(eval_env, config=cfg)
    sac_pink.load("models/sac/sac_pink")
    models["sac_pink"] = sac_pink

    sac_brown = SAC_HockeyAgent(eval_env, config=cfg)
    sac_brown.load("models/sac/sac_brown")
    models["sac_brown"] = sac_brown

    sac_reward = SAC_HockeyAgent(eval_env, config=cfg)
    sac_reward.load("models/sac/sac_reward")
    models["sac_reward"] = sac_reward

    sac_all_1 = SAC_HockeyAgent(eval_env, config=cfg)
    sac_all_1.load("models/sac/sac_all_1")
    models["sac_all_1"] = sac_all_1

    return models


def reset_env():
    """
    Fully resets Pygame and the environment to avoid rendering crashes.
    """
    pygame.quit()  # Quit Pygame
    pygame.display.quit()  # Close display
    gc.collect()  # Force garbage collection to clear old envs
    pygame.init()  # Reinitialize Pygame
    pygame.display.init()  # Reinitialize display


def eval_against_all_models(
    agent,
    models,
    eval_env,
    agent_name,
    num_episodes=10,
    save_path=None,
    render_mode="rgb_array",
):
    """
    Evaluate the agent against all saved models and generate performance plots.

    Args:
        agent: The agent to evaluate.
        models (dict): Dictionary of models to evaluate against.
        eval_env: The evaluation environment.
        agent_name (str): Name of the agent.
        num_episodes (int, optional): Number of episodes to evaluate. Defaults to 10.
        save_path (str, optional): Path to save the plot. Defaults to None.
        render_mode (str, optional): Render mode for evaluation. Defaults to "rgb_array".
    """
    print(
        f"\n ===== Evaluating *{agent_name}* against all models ({num_episodes} episodes) =====\n"
    )
    print("Specifically against:")
    for model_name, model in models.items():
        print(f"\t{model_name}: {model}")
    print("\n")

    # Collect evaluation data
    data = []
    for model_name, model in models.items():
        print(f"Evaluating {agent_name} VS {model_name}")

        # Force-reset environment before switching opponents
        eval_env.close()
        eval_env = None
        reset_env()  # Full reset of Pygame
        eval_env = h_env.HockeyEnv_BasicOpponent()  # Create fresh environment

        info = agent.evaluate(
            verbose=0,
            num_episodes=num_episodes,
            opponent_right=model,
            render_mode=render_mode,
            env=eval_env,
        )
        data.append(
            {
                "Model": model_name,
                "Mean Reward": info["mean_reward"],
                "Std Reward": info["std_reward"],
                "Win Rate": info["win_rate"],
                "Win Counts": info["win_counts"],
            }
        )

    # Create a DataFrame
    df = pd.DataFrame(data)

    print("==> Plotting ...")

    # Create a figure with two subplots
    fig, ax1 = plt.subplots(2, 1, figsize=(20, 16))

    # Background shading for Mean Reward (green above 0, red below 0)
    ax1[0].axhspan(0, max(df["Mean Reward"].max(), 0), facecolor="green", alpha=0.15)
    ax1[0].axhspan(min(df["Mean Reward"].min(), 0), 0, facecolor="red", alpha=0.15)

    # Set x positions as integers
    x_positions = range(len(df["Model"]))

    # Plot Mean Reward as points with Std Reward as error bars
    ax1[0].errorbar(
        x_positions,
        df["Mean Reward"],
        yerr=df["Std Reward"],
        fmt="o",
        capsize=5,
        label="Mean Reward ± Std",
        color="blue",
    )
    ax1[0].set_title(f"Mean Reward {agent_name}")
    ax1[0].set_ylabel("Mean Reward")
    ax1[0].set_xticks(x_positions)
    ax1[0].set_xticklabels(df["Model"], rotation=45, ha="right")
    ax1[0].legend()
    ax1[0].grid(True)

    # Annotate each point with its exact Mean Reward value
    for x, mean in zip(x_positions, df["Mean Reward"]):
        ax1[0].text(
            x,
            mean,
            f"{mean:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    # Background shading for Win Rate (green above 0.5, red below 0.5)
    ax1[1].axhspan(0.5, 1, facecolor="green", alpha=0.15)
    ax1[1].axhspan(0, 0.5, facecolor="red", alpha=0.15)

    # Plot Win Rate as a bar chart
    bars = ax1[1].bar(x_positions, df["Win Rate"], color="orange", label="Win Rate")

    total = 0
    for bar in bars:
        height = bar.get_height()
        total += height

    ax1[1].set_title(f"Win Rate {total}")
    ax1[1].set_ylabel("Win Rate")
    ax1[1].set_xticks(x_positions)
    ax1[1].set_xticklabels(df["Model"], rotation=45, ha="right")
    ax1[1].legend()
    ax1[1].grid(True)

    # Annotate each bar with its exact Win Rate value
    for bar in bars:
        height = bar.get_height()
        ax1[1].text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    # Adjust layout and show/save plot
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved at {save_path}")
    else:
        plt.show()

    # Print table of results
    print(df.to_string(index=False))
