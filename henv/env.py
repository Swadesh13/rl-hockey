from typing import List

import numpy as np
from gymnasium import spaces
from hockey import hockey_env
from stable_baselines3.common.env_util import make_vec_env

from henv.rewards import filter_reward, get_additional_rewards


class BasicOpponent(hockey_env.BasicOpponent):
    def __init__(self, weak=False, keep_mode=True):
        super().__init__(weak, keep_mode)

    def predict(self, state, deterministic=None):
        return self.act(state), None


class HockeyEnv_SB3(hockey_env.HockeyEnv):
    def __init__(
        self,
        weak_opponent: bool = False,
        additional_rewards: List[str] = None,
        reward_multiplier: float = 1.0,
    ):
        # Initialize the parent class with the weak_opponent parameter
        super().__init__()
        self.opponent = BasicOpponent(weak=weak_opponent)
        # linear force in (x,y)-direction, torque, and shooting
        self.action_space = spaces.Box(-1, +1, (4,), dtype=np.float32)

        # Check if additional_rewards contains only known reward types
        if additional_rewards:
            d = set(additional_rewards).difference(
                set(
                    [
                        "puck_throw_angle",
                        "pred_dist_from_puck",
                        "puck_infront",
                        "puck_intercept",
                        
                        "puck_positional",
                        "defensive_play",
                        "momentum_control",
                        "blocking",
                        
                        "puck_proximity",
                        "intercept_path",
                        "positional_control",
                        "defensive_coverage",
                        "offensive_pressure",
                    ]
                )
            )
            assert len(d) == 0, f"Unknown additional reward: {d}"
        self.additional_rewards = additional_rewards
        self.reward_multiplier = reward_multiplier
        print(
            f"Additional rewards: {additional_rewards}, Reward multiplier: {reward_multiplier}"
        )

    def step(self, action):
        ob2 = self.obs_agent_two()
        a2, _ = self.opponent.predict(ob2, deterministic=True)
        action2 = np.hstack([action, a2])
        # Perform the step in the parent class and get the observation, reward, done, time, and info
        obs, reward, done, t, info = super().step(action2)
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
        reward *= self.reward_multiplier
        return obs, reward, done, t, info

    def reset(self, seed=None, options=None, one_starting=None, mode=None):
        # Seed the environment with the given seed
        super().seed(seed)
        return super().reset(one_starting, mode)

    @staticmethod
    def make_vec_env(
        n_envs: int = 1,
        weak_opponent: bool = False,
        additional_rewards: List[str] = None,
        reward_multiplier: float = 1.0,
    ):
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
            return HockeyEnv_SB3(
                weak_opponent=weak_opponent,
                additional_rewards=additional_rewards,
                reward_multiplier=reward_multiplier,
            )

        return make_vec_env(create_env, n_envs=n_envs)
