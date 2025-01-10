import hockey.hockey_env as h_env
import numpy as np

from henv.env import BasicOpponent


def eval_agent(
    player_left, opponent_right=None, env=None, num_episodes=10, render_mode="human"
):
    """
    Runs an agent in a Hockey environment for a specified number of episodes.

    Parameters:
    - player_left: Trained agent to be tested.
    - opponent_right: Opponent in the environment. (default is BasicOpponent Strong)
    - env: The environment to run the agent in. (default is HockeyEnv)
    - num_episodes: Number of episodes to run the simulation (default is 10).
    - render_mode: Mode for rendering the environment (default is "human", other option "rgb_array").

    Returns:
    - mean_reward: Mean reward over all episodes.
    - std_reward: Standard deviation of rewards over all episodes.
    """
    if opponent_right is None:
        opponent_right = BasicOpponent(weak=False)

    if env is None:
        env = h_env.HockeyEnv()

    total_rewards = []

    for episode in range(num_episodes):
        obs, info = env.reset()
        obs_agent2 = env.obs_agent_two()
        episode_reward = 0

        while True:
            env.render(mode=render_mode)
            a1, _ = player_left.predict(obs, deterministic=True)
            a2, _ = opponent_right.predict(obs_agent2, deterministic=True)
            obs, reward, done, _, info = env.step(np.hstack([a1, a2]))
            obs_agent2 = env.obs_agent_two()
            episode_reward += reward

            if done:
                break

        total_rewards.append(episode_reward)
        print(f"Episode {episode + 1:<3} Reward: {episode_reward:>5.2f}")

    env.close()

    # Calculate and return mean reward and standard deviation
    mean_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    print(f"Overall Avg Reward: {mean_reward:>5.2f} ± {std_reward:.2f}")

    return mean_reward, std_reward
