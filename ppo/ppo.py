import numpy as np
import os
import sys
import torch
from stable_baselines3 import PPO
from utils.evaluate import eval_agent

class HockeyPPOAgent:
    def __init__(self, env, model_path="ppo/ppo_model", hyperparameters=None):
        """
        Initializes the PPO agent for the Hockey environment.

        Parameters:
        - env: The environment instance.
        - model_path: Path to save/load the model.
        - hyperparameters: Dictionary of PPO hyperparameters (optional).
        """
        self.env = env
        self.model_path = model_path
        self.hyperparameters = hyperparameters or {
            'learning_rate': 0.0003,
            'n_steps': 2048,
            'batch_size': 64,
            'n_epochs': 10,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_range': 0.2,
            'vf_coef': 0.5,
            'ent_coef': 0.01,
            'max_grad_norm': 0.5
        }

        self.model = PPO("MlpPolicy", self.env, **self.hyperparameters, verbose=1)

    def train(self, total_timesteps=100000):
        """
        Trains the PPO model.

        Parameters:
        - total_timesteps: Total timesteps for training.
        """
        print("Starting training...")
        self.model.learn(total_timesteps=total_timesteps)
        print("Training complete.")

    def evaluate(self, opponent_right=None, num_episodes=10, render_mode="human"):
        """
        Evaluates the trained PPO model in the environment.

        Parameters:
        - num_episodes: Number of episodes to evaluate.
        - render_mode: Mode for rendering the environment (default is "human").

        Returns:
        - mean_reward: Mean reward over all episodes.
        - std_reward: Standard deviation of rewards over all episodes.
        """
        return eval_agent(self.model, opponent_right=None, num_episodes=num_episodes, render_mode=render_mode)

    def save(self, save_path=None):
        """
        Saves the trained PPO model.
        """
        if save_path:
            path = save_path
        else:
            path = self.model_path
            
        self.model.save(path)
        print(f"Model saved at {path}.")

    def load(self, load_path=None):
        """
        Loads the PPO model.
        """
        if load_path:
            path = load_path
        else:
            path = self.model_path
            
        if os.path.exists(f"{path}.zip"):
            self.model = PPO.load(path, env=self.env)
            print(f"Model loaded from {path}.")
        else:
            print(f"No model found at {path}. Starting with a new model.")

if __name__ == "__main__":
    TRAIN = False
    EVAL = True
    
    from env import HockeyEnv_SB3
    env = HockeyEnv_SB3.make_vec_env(n_envs=1)
    agent = HockeyPPOAgent(env, model_path="ppo/ppo_hockey_model")
    
    if TRAIN:
        agent.train(total_timesteps=100000)
        agent.save()
        
    if EVAL:
        agent.load()
        agent.evaluate(num_episodes=5)
        
