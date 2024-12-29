# Custom rewards based on current obs
import numpy as np


def filter_reward(obs, reward):
    """
    There is a situation where the reward is -ve even though the agent has the puck -
    due to some inherent movement of the puck (when the agent goes back)
    """
    if obs[-2]:
        return 0
    return reward


def puck_throw_angle(obs, h_env):
    # consider elastic collisions with constant velocity - calculate if at present angle puck can be shot
    x1, y1, a1 = abs(obs[0]), obs[1], obs[2]
    if obs[16]:
        puck_future_pos = y1 + (x1 + h_env.W / 2) * np.tan(a1) - np.arange(3) * h_env.H
        goal_size = h_env.GOAL_SIZE / h_env.SCALE
        if np.any(np.abs(puck_future_pos) < goal_size):
            return 0.5
        else:
            return -0.1
    return 0


def pred_distance_from_puck(obs):
    if obs[14] < 0 and obs[12] > 0 and obs[-1] == 0:
        m = obs[15] / obs[14]
        a = -m
        c = m * obs[12] - obs[13]
        if a * -1 + 3 + c > 0 and a * -1 + -3 + c < 0:  # only if puck is predicted to be in this region
            d = abs(a * obs[0] + obs[1] + c) / (m**2 + 1)
            return -d / 8
    return 0


def puck_infront(obs):
    if obs[12] < obs[0] - 0.2:
        return -0.5
    return 0


def get_additional_rewards(obs, h_env):
    rewards = {}
    rewards["puck_throw_angle"] = puck_throw_angle(obs, h_env)
    rewards["pred_dist_from_puck"] = pred_distance_from_puck(obs)
    rewards["puck_infront"] = puck_infront(obs)
    return rewards
