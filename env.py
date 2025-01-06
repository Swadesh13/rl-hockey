from typing import List
from hockey import hockey_env
from hyperparams import REWARD_MULTIPLIER
from rewards import filter_reward, get_additional_rewards
from stable_baselines3.common.vec_env import DummyVecEnv


class HockeyEnv_SB3(hockey_env.HockeyEnv_BasicOpponent):
    def __init__(self, weak_opponent=False, additional_rewards: List[str] = None):
        # Initialize the parent class with the weak_opponent parameter
        super().__init__(weak_opponent=weak_opponent)
        # Check if additional_rewards contains only known reward types
        if additional_rewards:
            assert (
                len(set(additional_rewards).difference(set(["puck_throw_angle", "pred_dist_from_puck", "puck_infront"]))) == 0
            ), "Unknown additional reward"
        self.additional_rewards = additional_rewards

    def step(self, action):
        # Perform the step in the parent class and get the observation, reward, done, time, and info
        obs, reward, done, t, info = super().step(action)
        # Filter the reward using the filter_reward function
        reward = filter_reward(obs, reward)
        # Add the reward for touching the puck
        reward += info["reward_touch_puck"]
        # If additional rewards are specified, add them to the reward
        if self.additional_rewards:
            r2 = get_additional_rewards(obs, hockey_env)
            for key in self.additional_rewards:
                reward += r2[key]
        # Multiply the reward by the reward multiplier
        reward *= REWARD_MULTIPLIER
        return obs, reward, done, t, info

    def reset(self, seed=None, options=None, one_starting=None, mode=None):
        # Seed the environment with the given seed
        super().seed(seed)
        return super().reset(one_starting, mode)
    
    @staticmethod
    def make_vec_env(n_envs=1, weak_opponent=False, additional_rewards: List[str] = None):
        """
        Create and return a vectorized version of the HockeyEnv_SB3 environment.

        Args:
            n_envs (int): Number of environments to vectorize.
            weak_opponent (bool): Whether to use a weak opponent.
            additional_rewards (List[str]): List of additional rewards to use.

        Returns:
            VecEnv: A vectorized environment.
        """
        def create_env():
            return HockeyEnv_SB3(weak_opponent=weak_opponent, additional_rewards=additional_rewards)

        return make_vec_env(create_env, n_envs=n_envs)