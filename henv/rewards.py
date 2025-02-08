# Custom rewards based on current obs
import math

import numpy as np


class Henv:
    FPS = 50
    SCALE = 60.0  # affects how fast-paced the game is, forces should be adjusted as well (Don't touch)

    VIEWPORT_W = 600
    VIEWPORT_H = 480
    W = VIEWPORT_W / SCALE
    H = VIEWPORT_H / SCALE
    CENTER_X = W / 2
    CENTER_Y = H / 2
    ZONE = W / 20
    MAX_ANGLE = math.pi / 3  # Maximimal angle of racket
    MAX_TIME_KEEP_PUCK = 15
    GOAL_SIZE = 75

    RACKETPOLY = [
        (-10, 20),
        (+5, 20),
        (+5, -20),
        (-10, -20),
        (-18, -10),
        (-21, 0),
        (-18, 10),
    ]
    RACKETFACTOR = 1.2

    FORCEMULTIPLIER = 6000
    SHOOTFORCEMULTIPLIER = 60
    TORQUEMULTIPLIER = 400
    MAX_PUCK_SPEED = 25


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


def puck_intercept(obs):
    # consider elastic collisions with constant velocity - calculate if at present angle puck can be shot
    if obs[16]:
        x1, y1, a1 = obs[0], obs[1], obs[2]
        x2, y2 = obs[6], obs[7]
        d = abs((x2 - x1) * np.sin(a1) - (y2 - y1) * np.cos(a1))
        if d < 1:
            return d - 1
    return 0


def pred_distance_from_puck(obs):
    if obs[14] < 0 and obs[12] > 0 and obs[-1] == 0:
        m = obs[15] / obs[14]
        a = -m
        c = m * obs[12] - obs[13]
        if (
            a * -1 + 3 + c > 0 and a * -1 + -3 + c < 0
        ):  # only if puck is predicted to be in this region
            d = abs(a * obs[0] + obs[1] + c) / (m**2 + 1)
            return -d / 8
    return 0


def puck_infront(obs):
    if obs[12] < obs[0] - 0.2:
        return -0.5
    return 0


# ================== PPO's rewards ==================


def is_puck_between_player_and_goal(player_x, puck_x, threshold=0.3):
    """ Checks if the player is between the puck and the goal, considering a threshold. """
    return player_x < puck_x + threshold


def defensive_play(obs, h_env):
    """
    Combines multiple reward functions to create a well-rounded defensive reward.
    - Encourages staying between the puck and the goal.
    - Rewards positioning near the puck's predicted trajectory.
    - Encourages blocking and defensive interception.
    """

    return (
        0.4 * reward_intercept_path(obs) +     # Predictive positioning
        0.3 * reward_puck_proximity(obs) +     # Staying close to the puck
        0.2 * blocking(obs, h_env) +           # Reward for interception
        0.1 * reward_puck_between_player_and_goal(obs)    # Penalty if puck not between player & goal
    )


def puck_positional(obs, h_env):
    """
    Combines multiple reward functions to create a well-rounded offensive positioning reward.
    - Encourages staying close to the puck.
    - Encourages being between the puck and the opponent’s goal.
    - Encourages moving forward toward the goal.
    """

    # Weighted combination of relevant reward functions
    return (
        0.4 * reward_puck_proximity(obs) +           # Staying close to the puck
        0.3 * reward_puck_between_player_and_goal(obs) + # Being between puck & opponent goal
        0.3 * reward_offensive_pressure(obs, h_env)   # Moving toward the opponent's goal
    )


def momentum_control(obs): # okay ish score
    linear_speed = np.linalg.norm(obs[3:5])  # Player linear velocity
    angular_speed = abs(obs[5])  # Player angular velocity

    # Smooth penalty that starts low and increases gradually
    penalty = -0.02 * (linear_speed - 5) ** 2 - 0.02 * (angular_speed - 2) ** 2

    # Reward for maintaining controlled movement in an optimal range
    if 3 < linear_speed < 8 and 1 < angular_speed < 4:
        reward = 0.3  # Reward for smooth controlled movement
    else:
        reward = penalty  # Apply smooth penalty

    return reward


def blocking(obs, h_env): #gut
    """
    Reward for intercepting the puck near the agent's own goal.
    """
    puck_x, puck_y, puck_vx = obs[12], obs[13], obs[14]
    player_x, player_y = obs[0], obs[1]

    # Reward if the agent is near the puck and the puck is heading toward its goal
    if puck_vx < 0:
        dist_to_puck = np.sqrt((puck_x - player_x) ** 2 + (puck_y - player_y) ** 2)
        if dist_to_puck < 0.5:
            return 0.3
    return 0


def reward_puck_proximity(obs):
    """ Encourages staying close to the puck. """
    player_x, player_y = obs[0], obs[1]
    puck_x, puck_y = obs[12], obs[13]

    distance_to_puck = np.sqrt((player_x - puck_x) ** 2 + (player_y - puck_y) ** 2)

    # Scale reward to be between 0 and -0.5 (penalty for being far)
    return max(-distance_to_puck * 0.02, -0.5)


def reward_intercept_path(obs):
    """ Rewards the agent for positioning itself close to the predicted puck trajectory. """
    
    # Unpacking relevant observation values
    agent_x, agent_y = obs[0], obs[1]  # Agent's position
    puck_x, puck_y = obs[12], obs[13]  # Puck's position
    puck_vx, puck_vy = obs[14], obs[15]  # Puck's velocity
    game_state = obs[-1]  # Game state indicator

    # Condition: Puck must be moving towards the agent (left direction) and game must be in normal state
    if puck_vx < 0 and puck_x > 0 and game_state == 0:
        
        # Compute the slope of the puck's trajectory (avoiding division by zero)
        if puck_vx != 0:
            trajectory_slope = puck_vy / puck_vx  
        else:
            trajectory_slope = 0  # If puck is not moving in x, assume flat trajectory

        # Compute the y-intercept of the puck's trajectory: y = mx + b → b = y - mx
        trajectory_intercept = puck_y - trajectory_slope * puck_x

        # Find the agent's **perpendicular distance** from the predicted puck trajectory
        numerator = abs(trajectory_slope * agent_x - agent_y + trajectory_intercept)
        denominator = (trajectory_slope**2 + 1) ** 0.5
        perpendicular_distance = numerator / denominator

        # Penalize the distance (higher distance → larger penalty)
        return max(-perpendicular_distance / 8, -0.5)  

    return 0  # No reward if conditions are not met


def reward_puck_between_player_and_goal(obs):
    """ Rewards the agent for staying between the puck and the opponent's goal. """
    player_x = obs[0]
    puck_x = obs[12]

    if is_puck_between_player_and_goal(player_x, puck_x):  # Puck is between player and goal
        return 0
    return -0.5  # penalty if not in a good position


def reward_offensive_pressure(obs, h_env):
    """ Encourages the agent to stay near the opponent’s goal to apply pressure. """
    player_x = obs[0]
    goal_x = h_env.W  # Opponent's goal

    return max(0.3 - abs(player_x - goal_x) * 0.1, 0)  # Reward for being near the opponent’s goal


# ====================== End of Vojtech's rewards ======================


def get_additional_rewards(obs, h_env=Henv):
    rewards = {}
    rewards["puck_throw_angle"] = puck_throw_angle(obs, h_env)
    rewards["pred_dist_from_puck"] = pred_distance_from_puck(obs)
    rewards["puck_infront"] = puck_infront(obs)
    rewards["puck_intercept"] = puck_intercept(obs)
    
    rewards["puck_positional"] = puck_positional(obs, h_env)
    rewards["defensive_play"] = defensive_play(obs, h_env)
    rewards["momentum_control"] = momentum_control(obs)
    rewards["blocking"] = blocking(obs, h_env)
    
    rewards["puck_proximity"] = reward_puck_proximity(obs)
    rewards["intercept_path"] = reward_intercept_path(obs)
    rewards["puck_between_player_and_goal"] = reward_puck_between_player_and_goal(obs)
    rewards["offensive_pressure"] = reward_offensive_pressure(obs, h_env)
    
    return rewards
