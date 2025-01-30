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


# ================== Vojtech's rewards ==================


def is_between_puck_and_goal(player_x, puck_x, goal_x):
    """
    Checks if the player is positioned between the puck and the goal.
    works for both players if needed for league
    - Returns True if the player is between the puck and their goal.
    """
    return (goal_x - puck_x) * (player_x - puck_x) < 0


def defensive_play(obs):
    """
    Improved defensive play reward:
    - Considers puck trajectory with possible wall bounces.
    - Checks if the player is between the puck and goal, even after a reflection.
    """
    player_x, player_y = obs[0], obs[1]
    puck_x, puck_y, puck_vx, puck_vy = obs[12], obs[13], obs[14], obs[15]
    goal_x = 0 if puck_x < Henv.CENTER_X else Henv.W  # Own goal position
    wall_top = Henv.H / 2  # Top boundary
    wall_bottom = -Henv.H / 2  # Bottom boundary

    # 1. Check if player is between puck and goal (X-direction)
    is_between_x = is_between_puck_and_goal(player_x, puck_x, goal_x)

    # 2. Compute expected puck trajectory (initial)
    if puck_vx != 0:
        trajectory_slope = puck_vy / puck_vx  # dy/dx
        trajectory_intercept = puck_y - trajectory_slope * puck_x  # y = mx + b

        # 3. Check for wall reflection
        if puck_vy > 0:  # Moving upward, may hit the top wall
            time_to_wall = (
                wall_top - puck_y
            ) / puck_vy  # Time until hitting the top wall
            if time_to_wall > 0:
                puck_x += puck_vx * time_to_wall
                puck_y = wall_top
                puck_vy = -puck_vy  # Reflect downward
        elif puck_vy < 0:  # Moving downward, may hit the bottom wall
            time_to_wall = (wall_bottom - puck_y) / puck_vy
            if time_to_wall > 0:
                puck_x += puck_vx * time_to_wall
                puck_y = wall_bottom
                puck_vy = -puck_vy  # Reflect upward

        # 4. Compute new trajectory after bounce
        trajectory_slope = puck_vy / puck_vx
        trajectory_intercept = puck_y - trajectory_slope * puck_x
        expected_puck_y = (
            trajectory_slope * player_x + trajectory_intercept
        )  # Where puck will be at player_x

        # 5. Check if player is near this trajectory
        y_tolerance = 0.3  # Allow some leeway
        is_on_trajectory = abs(player_y - expected_puck_y) < y_tolerance
    else:
        is_on_trajectory = (
            False  # No x velocity, meaning no strong trajectory to predict
        )

    # 6. Reward based on positioning and trajectory prediction
    if is_between_x and is_on_trajectory:
        return 0.6  # Strong reward for good positioning even after bounce
    elif is_between_x:
        return 0.3  # Some reward for being in the right x-region
    else:
        return (
            -0.4
        )  # Penalty if puck moves toward the goal and player is out of position


def puck_positional(obs, h_env):
    """
    Reward for maintaining a strategic position relative to the puck and goal.
    - Encourages being between puck and opponent goal.
    """
    player_x, player_y = obs[0], obs[1]
    puck_x, puck_y = obs[12], obs[13]
    goal_x = h_env.W if puck_x > h_env.CENTER_X else 0  # Opponent's goal

    # Encourage proximity to the puck
    dist_to_puck = np.sqrt((player_x - puck_x) ** 2 + (player_y - puck_y) ** 2)
    positional_reward = -dist_to_puck * 0.05  # Reduced scaling

    # Reward for staying between puck and goal
    between_puck_goal = is_between_puck_and_goal(player_x, puck_x, goal_x)
    positional_reward += between_puck_goal * 0.8  # Increased impact

    return positional_reward


def momentum_control(obs):
    linear_speed = np.linalg.norm(obs[3:5])  # Player linear velocity
    angular_speed = abs(obs[5])  # Player angular velocity

    # Use a smooth quadratic penalty instead of a fixed penalty
    penalty = -0.05 * (linear_speed**2 + angular_speed**2)

    # Reward controlled movement rather than absolute speed constraints
    reward = 0.2 if linear_speed < 7 and angular_speed < 3 else penalty

    return reward


def blocking(obs, h_env):
    """
    Reward for intercepting the puck near the agent's own goal.
    """
    puck_x, puck_y, puck_vx = obs[12], obs[13], obs[14]
    goal_x = 0 if puck_x < h_env.CENTER_X else h_env.W
    player_x, player_y = obs[0], obs[1]

    # Reward if the agent is near the puck and the puck is heading toward its goal
    if (goal_x == 0 and puck_vx < 0) or (goal_x == h_env.W and puck_vx > 0):
        dist_to_puck = np.sqrt((puck_x - player_x) ** 2 + (puck_y - player_y) ** 2)
        if dist_to_puck < 0.5:
            return 0.3
    return 0


# ====================== End of Vojtech's rewards ======================


def get_additional_rewards(obs, h_env=Henv):
    rewards = {}
    rewards["puck_throw_angle"] = puck_throw_angle(obs, h_env)
    rewards["pred_dist_from_puck"] = pred_distance_from_puck(obs)
    rewards["puck_infront"] = puck_infront(obs)
    rewards["puck_intercept"] = puck_intercept(obs)
    rewards["puck_positional"] = puck_positional(obs, h_env)
    rewards["defensive_play"] = defensive_play(obs)
    rewards["momentum_control"] = momentum_control(obs)
    rewards["blocking"] = blocking(obs, h_env)
    return rewards
