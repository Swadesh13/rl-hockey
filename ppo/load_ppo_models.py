from ppo.ppo import PPO_HockeyAgent
from utils.parsing import convert_to_cfgnode, get_eval_env, load_config


def load_ppo_agent(config_path: str):
    """
    Load a PPO agent from a saved model.

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
    Load all saved PPO agents.
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


import matplotlib.pyplot as plt
import pandas as pd


def eval_against_all_models(agent, models, eval_env, agent_name, num_episodes=10):
    """
    Evaluate the agent against all saved models and plot Mean Reward with Std Error Bars
    while also displaying Win Rate as a bar chart.

    Parameters:
    - agent: The agent to evaluate.
    - num_episodes: Number of episodes to evaluate.
    - models: dictionary of models to evaluate against. {name: agent}
    """
    print(
        f"\n ===== Evaluating *{agent_name}* against all models ({num_episodes} episodes) =====\n"
    )

    # Collect evaluation data
    data = []
    for model_name, model in models.items():
        print(f"Evaluating {agent_name} VS {model_name}")
        info = agent.evaluate(
            verbose=0,
            num_episodes=num_episodes,
            opponent_right=model,
            render_mode="rgb_array",
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
    fig, ax1 = plt.subplots(2, 1, figsize=(10, 8))

    # Background shading for Mean Reward (green above 0, red below 0)
    ax1[0].axhspan(0, max(df["Mean Reward"].max(), 0), facecolor="green", alpha=0.15)
    ax1[0].axhspan(min(df["Mean Reward"].min(), 0), 0, facecolor="red", alpha=0.15)

    # Plot Mean Reward as points with Std Reward as error bars
    ax1[0].errorbar(
        df["Model"],
        df["Mean Reward"],
        yerr=df["Std Reward"],
        fmt="o",
        capsize=5,
        label="Mean Reward ± Std",
        color="blue",
    )
    ax1[0].set_title(f"Mean Reward {agent_name}")
    ax1[0].set_ylabel("Mean Reward")
    ax1[0].set_xticks(range(len(df["Model"])))
    ax1[0].set_xticklabels(df["Model"], rotation=45, ha="right")
    ax1[0].legend()
    ax1[0].grid(True)

    # Background shading for Win Rate (green above 0.5, red below 0.5)
    ax1[1].axhspan(0.5, 1, facecolor="green", alpha=0.15)
    ax1[1].axhspan(0, 0.5, facecolor="red", alpha=0.15)

    # Plot Win Rate as a bar chart
    ax1[1].bar(df["Model"], df["Win Rate"], color="orange", label="Win Rate")
    ax1[1].set_title("Win Rate")
    ax1[1].set_ylabel("Win Rate")
    ax1[1].set_xticks(range(len(df["Model"])))
    ax1[1].set_xticklabels(df["Model"], rotation=45, ha="right")
    ax1[1].legend()
    ax1[1].grid(True)

    # Adjust layout and show plot
    plt.tight_layout()
    plt.show()

    # Print table of results
    print(df.to_string(index=False))
