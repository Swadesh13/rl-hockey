import os

import numpy as np
from stable_baselines3 import TD3
import gymnasium as gym
from stable_baselines3.common.noise import NormalActionNoise


from utils.evaluate import eval_agent
from utils.parsing import *


class PendulumTD3Agent:
    def __init__(self, env, config):
        """
        Initializes the TD3 agent for the Hockey environment.

        Parameters:
        - env: The environment instance.
        - config: Configuration node (CfgNode).
        """
        self.env = env
        self.config = config
        if not self.config.training.model_path:
            raise ValueError(
                "Model path for training is not specified in the configuration."
            )
        self.model_path = self.config.training.model_path

        hyperparameters = self.config.model.hyperparameters

        action_noise = None
        if self.config.model.action_noise is not None and self.config.model.action_noise.type == "normal":
          n_actions = env.action_space.shape[-1]
          action_noise = NormalActionNoise(mean = np.full(n_actions, self.config.model.action_noise.mean), 
                                                          sigma= self.config.model.action_noise.sigma * np.ones(n_actions))
    
        self.model = TD3(
            action_noise=action_noise,
            **hyperparameters,
            verbose=self.config.logging.verbose,
        )

    def train(self, total_timesteps=None):
        """
        Trains the TD3 model.

        Parameters:
        - total_timesteps: Total timesteps for training.
        """
        tt = total_timesteps or self.config.training.total_timesteps

        print("Starting training...")
        self.model.learn(total_timesteps=tt,progress_bar=True)
        print("Training complete.")

    def evaluate(self, num_episodes=10, render_mode="human"):
        return self.eval_agent()

    def eval_agent(self, num_episodes=10,max_steps=200, render_mode="human"):
      """
      Evaluates the trained TD3 model in the specified environment.

      Parameters:
      - model: Trained TD3 agent to evaluate.
      - env: The environment to run the agent in.
      - num_episodes: Number of episodes to run the evaluation.
      - render_mode: Mode for rendering the environment (default is "human").

      Returns:
      - mean_reward: Mean reward over all episodes.
      - std_reward: Standard deviation of rewards over all episodes.
      """
      total_rewards = []
      max_stagnant_steps = 5  # Adjust this value based on your environment
      for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0

        previous_reward = None
        stagnant_steps = 0

        for i in range(max_steps):
          env.render()
            
          action, _ = self.model.predict(obs, deterministic=True)
          obs, reward, done, _, _ = env.step(action)
          episode_reward += reward

          if reward == previous_reward:
              stagnant_steps += 1
          else:
              stagnant_steps = 0  

          previous_reward = reward

          if stagnant_steps >= max_stagnant_steps:
                print(f"Breaking out due to stagnant reward after {stagnant_steps} steps")
                break

          if done:
            break
        total_rewards.append(episode_reward)
        print(f"Episode {episode + 1:<3} Reward: {episode_reward:>5.2f}")

      mean_reward = np.mean(total_rewards)
      std_reward = np.std(total_rewards)
      print(f"Evaluation complete. Mean Reward: {mean_reward:.2f}, Std Reward: {std_reward:.2f}")
      return mean_reward, std_reward

    def save(self, save_path=None):
        """
        Saves the trained TD3 model.
        """
        path = save_path or self.model_path
        self.model.save(path)
        print(f"Model saved at {path}")

    def load(self, load_path=None):
        """
        Loads the TD3 model.
        """
        path = load_path or self.model_path
        if os.path.exists(f"{path}.zip"):
            self.model = TD3.load(path, env=self.env)
            print(f"Model loaded from {path}")
        else:
            print(f"No model found at {path}. Starting with a new model.")


if __name__ == "__main__":
    args = parse_args()
    cfg = get_config_from_args(args, cfgnode=True)

    if not args.quiet:
        print_config(cfg)
        print_args(args)

    env = gym.make(cfg.environment.env_name,render_mode=cfg.environment.render_mode)

    agent = PendulumTD3Agent(env, config=cfg)

    if args.train:
        agent.train(total_timesteps=cfg.training.total_timesteps)
        agent.save()

    if args.eval:
        agent.load()
        agent.evaluate(num_episodes=args.eval_episodes)