import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Callable, Union
from fvcore.common.config import CfgNode
import gymnasium as gym
from henv.env import HockeyEnv_SB3

def CreateExplorationNoise(config: CfgNode) -> Callable[[Union[torch.Tensor, List[torch.Tensor]]], Union[torch.Tensor, List[torch.Tensor]]]:
    if config.type == "normal":
        return lambda Action: (
            [torch.randn_like(A) * config.magnitude for A in Action]
            if isinstance(Action, list) else torch.randn_like(Action) * config.magnitude
        )

    elif config.type == "uniform":
        return lambda Action: (
            [(torch.rand_like(A) * 2 - 1) * config.magnitude for A in Action]
            if isinstance(Action, list) else (torch.rand_like(Action) * 2 - 1) * config.magnitude
        )

    else:
        raise ValueError(f"Invalid exploration noise type: {config.type}")


def CreateHoeckyEnvs(conf : CfgNode) -> gym.vector.AsyncVectorEnv:
    envs = gym.vector.AsyncVectorEnv(
        [_CreateHockyEnvWrapper(conf, i) for i in range(conf.n_envs)]
    )
    return envs
    
def CreateHoeckyEnv(conf : CfgNode, seedExtra : int = 0) -> HockeyEnv_SB3:
    env = HockeyEnv_SB3(
        weak_opponent = conf.weak_opponent,
        additional_rewards = conf.additional_rewards,
        reward_multiplier = conf.reward_multiplier
    )
    env.seed(conf.seed + seedExtra)
    return env

def _CreateHockyEnvWrapper(conf : CfgNode, seedExtra : int = 0) -> Callable:
    def _init():
        return CreateHoeckyEnv(conf, seedExtra) 
    return _init