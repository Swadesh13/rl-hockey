# This file is for visualizing and loading tournament data for visualization or training purposes.

from hockey import hockey_env
import os
import glob
import numpy as np
import cv2
from typing import List
from henv.rewards import filter_reward, get_additional_rewards


class DummyHockeyEnv_SB3(hockey_env.HockeyEnv):
    def __init__(
        self,
        additional_rewards: List[str] = None,
        reward_multiplier: float = 1.0,
    ):
        # Initialize the parent class with the weak_opponent parameter
        super().__init__()
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
                        "puck_between_player_and_goal",
                        "offensive_pressure",
                    ]
                )
            )
            assert len(d) == 0, f"Unknown additional reward: {d}"
        self.additional_rewards = additional_rewards
        self.reward_multiplier = reward_multiplier
        # print(
        #     f"Additional rewards: {additional_rewards}, Reward multiplier: {reward_multiplier}"
        # )

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
        reward *= self.reward_multiplier
        info["TimeLimit.truncated"] = False
        return obs, reward, done, t, info

    def reset(self, seed=None, options=None, one_starting=None, mode=None):
        # Seed the environment with the given seed
        super().seed(seed)
        return super().reset(one_starting, mode)


def load_env_obs(env: hockey_env.HockeyEnv, obs):
    """
    Given the env and observation at any step, load the env with the observation.
    """
    CENTER_X, CENTER_Y = hockey_env.CENTER_X, hockey_env.CENTER_Y

    env.player1.position = [obs[0] + CENTER_X, obs[1] + CENTER_Y]
    env.player1.angle = obs[2]
    env.player1.linearVelocity = [obs[3], obs[4]]
    env.player1.angularVelocity = obs[5]
    env.player2.position = [obs[6] + CENTER_X, obs[7] + CENTER_Y]
    env.player2.angle = obs[8]
    env.player2.linearVelocity = [obs[9], obs[10]]
    env.player2.angularVelocity = obs[11]
    env.puck.position = [obs[12] + CENTER_X, obs[13] + CENTER_Y]
    env.puck.linearVelocity = [obs[14], obs[15]]
    if env.keep_mode:
        env.player1_has_puck = obs[16]
        env.player2_has_puck = obs[17]


def create_video(source, fps=30, output_name="output"):
    out = cv2.VideoWriter(
        output_name + ".mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (source[0].shape[1], source[0].shape[0]),
    )
    for i in range(len(source)):
        out.write(source[i])
    out.release()


def load_episode_and_save_video(
    ep_path,
    env: hockey_env.HockeyEnv = DummyHockeyEnv_SB3(),
    key="observations_round_0",
):
    """
    Given an episode, create a video of the match.
    """
    data = np.load(ep_path, allow_pickle=True)
    frames = []
    for obs in data[key]:
        load_env_obs(env, obs)
        frames.append(env.render(mode="rgb_array")[..., ::-1])
    create_video(frames, 30, key)


def env_obs_action_step(env, obs, action, bi_dir=True):
    """
    Collect one step for given env, obs, action.

    bi_dir: bool, if True, collect for the reversed board (i.e. player1 -> player2, player2 -> player1)
    """
    _idx = 3 if not env.keep_mode else 4
    load_env_obs(env, obs)
    if bi_dir:
        opp_obs = env.obs_agent_two()
    next_obs, reward, done, _, info = env.step(action)
    data = [
        [
            obs[None,],
            next_obs[None,],
            action[:_idx][None,],
            np.array([reward]),
            np.array([done]),
            [info],
        ]
    ]
    if bi_dir:
        load_env_obs(env, opp_obs)
        opp_action = np.hstack([action[_idx:], action[:_idx]])
        next_obs, reward, done, _, info = env.step(opp_action)
        data.append(
            [
                opp_obs[None,],
                next_obs[None,],
                opp_action[:_idx][None,],
                np.array([reward]),
                np.array([done]),
                [info],
            ]
        )
    return data


def load_episodes_to_replay_buffer(
    ep_path,
    replay_buffer,
    env: hockey_env.HockeyEnv = DummyHockeyEnv_SB3(),
):
    """
    Load one episode to a given replay buffer.
    """
    data = np.load(ep_path, allow_pickle=True)
    for key in data.keys():
        if "observations" in key:
            for obs, action in zip(
                data[key][:-1], data[key.replace("observations", "actions")]
            ):
                replay_data = env_obs_action_step(env, obs, action)
                for d in replay_data:
                    replay_buffer.add(*d)


def load_all_episodes_to_replay_buffer(
    ep_dir,
    replay_buffer,
    env: hockey_env.HockeyEnv = DummyHockeyEnv_SB3(),
):
    """Load all episodes in directory to buffer."""
    files = glob.glob(os.path.join(ep_dir, "*.pkl"))
    for f in files:
        load_episodes_to_replay_buffer(f, replay_buffer, env)


if __name__ == "__main__":
    # load_episode_and_save_video("cbdccad8-440f-4f82-b2d3-8f22666e6307.pkl")
    from henv.env import HockeyEnv_SB3
    from utils.memory import ExperienceMemory
    from henv.utils import load_all_episodes_to_replay_buffer

    env = HockeyEnv_SB3()
    buffer = ExperienceMemory(50000, env.observation_space, env.action_space)
    load_all_episodes_to_replay_buffer("rl-game-data", buffer, env)
