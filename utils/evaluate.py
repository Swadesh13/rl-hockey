import hockey.hockey_env as h_env
import numpy as np
import pygame

from henv.env import BasicOpponent


def eval_agent(
    player_left,
    opponent_right=None,
    env=None,
    num_episodes=10,
    render_mode="human",
    modes=["NORMAL"],
    verbose=2,
):
    """
    Runs an agent in a Hockey environment for a specified number of episodes.

    Parameters:
    - player_left: Trained agent to be tested.
    - opponent_right: Opponent in the environment. (default is BasicOpponent Strong)
    - env: The environment to run the agent in. (default is HockeyEnv)
    - num_episodes: Number of episodes to run the simulation (default is 10).
    - render_mode: Mode for rendering the environment (default is "human", other option "rgb_array").
    - modes: List of modes to cycle through during evaluation (e.g., ["NORMAL", "TRAIN_SHOOTING", "TRAIN_DEFENSE"]).

    Returns:
    - mean_reward: Mean reward over all episodes.
    - std_reward: Standard deviation of rewards over all episodes.
    """
    # print("eval_agent")

    if opponent_right is None:
        print("No opponent provided. Using BasicOpponent Strong.")
        opponent_right = BasicOpponent(weak=False)

    if env is None:
        print("No environment provided. Using HockeyEnv.")
        env = h_env.HockeyEnv_BasicOpponent()

    if not pygame.display.get_init():
        print("Reinitializing pygame display...")
        pygame.display.init()
        # env.screen = pygame.display.set_mode((800, 600))  # Adjust if needed

    if verbose > 0:
        print(f"==> {player_left=} VS {opponent_right} <==")
        # print(f"(weak={opponent_right.weak})")
        # print(f"{num_episodes=} {render_mode=} {modes=}")

    total_rewards = []
    win_counts = {"Agent Wins": 0, "Opponent Wins": 0, "Draws": 0}
    episodes_per_mode = num_episodes // len(modes)

    if verbose > 0:
        print(f"Running {num_episodes} episodes in {modes} modes...")

    for episode in range(num_episodes):
        # Determine the mode based on the episode range
        mode_idx = episode // episodes_per_mode
        if mode_idx >= len(modes):
            mode_idx = len(modes) - 1
        mode = modes[mode_idx]

        obs, info = env.reset(mode=mode)  # Reset environment with the new mode
        # print(f"Starting Episode {episode + 1} in Mode: {mode}")

        obs_agent2 = env.obs_agent_two()
        episode_reward = 0

        while True:
            if not pygame.display.get_init():
                print("Reinitializing pygame display...")
                pygame.display.init()
                # env.screen = pygame.display.set_mode((800, 600))  # Adjust if needed

            env.render(mode=render_mode)
            a1, _ = player_left.predict(obs, deterministic=True)
            a2, _ = opponent_right.predict(obs_agent2, deterministic=True)
            obs, reward, done, _, info = env.step(np.hstack([a1, a2]))
            obs_agent2 = env.obs_agent_two()
            episode_reward += reward

            if done:
                break

        total_rewards.append(episode_reward)

        # Determine winner
        if info["winner"] == 1:
            win_counts["Agent Wins"] += 1
        elif info["winner"] == -1:
            win_counts["Opponent Wins"] += 1
        else:
            win_counts["Draws"] += 1

        if verbose > 1:
            print(
                f"Episode {episode + 1:<3} Reward: {episode_reward:>6.2f} | Winner: {get_winner_name(info['winner'])}"
            )

    env.close()

    # Calculate and return mean reward and standard deviation
    mean_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    win_rate = win_counts["Agent Wins"] / num_episodes
    if verbose > 0:
        print(f"Overall Avg Reward: {mean_reward:>5.2f} ± {std_reward:.2f}")
        print(f"Win Statistics: {win_counts} win_rate={win_rate*100:.2f}%")

    info = {
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "win_counts": win_counts,
        "win_rate": win_rate,
    }

    return info


def get_winner_name(winner):
    if winner == 1:
        return "Left (Agent)"
    elif winner == -1:
        return "Right (Opponent)"
    elif winner == 0:
        return "Draw"
    else:
        return "Unknown"
