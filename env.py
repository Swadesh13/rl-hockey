from typing import List
from hockey import hockey_env
from hyperparams import REWARD_MULTIPLIER
from rewards import filter_reward, get_additional_rewards


class HockeyEnv_SB3(hockey_env.HockeyEnv_BasicOpponent):
    def __init__(self, weak_opponent=False, additional_rewards: List[str] = None):
        super().__init__(weak_opponent=weak_opponent)
        if additional_rewards:
            assert (
                len(set(additional_rewards).difference(set(["puck_throw_angle", "pred_dist_from_puck", "puck_infront"]))) == 0
            ), "Unknown additional reward"
        self.additional_rewards = additional_rewards

    def step(self, action):
        obs, reward, done, t, info = super().step(action)
        reward = filter_reward(obs, reward)
        reward += info["reward_touch_puck"]
        if self.additional_rewards:
            r2 = get_additional_rewards(obs, hockey_env)
            for key in self.additional_rewards:
                reward += r2[key]
        reward *= REWARD_MULTIPLIER
        return obs, reward, done, t, info

    def reset(self, seed=None, options=None, one_starting=None, mode=None):
        super().seed(seed)
        return super().reset(one_starting, mode)
