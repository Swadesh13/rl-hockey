from typing import List, Callable
import gymnasium as gym
from henv.env import HockeyEnv_SB3
from henv.hockey_agent import HockeyAgent
from utils.load import LoadTD7Agents,LoadAllPPOAgents,LoadAllSacAgents
from gymnasium.wrappers import TimeLimit
from henv.env import BasicOpponent
import os
from glob import glob

def CreateHockyEnvVector(envNum : int = 16,seed : int = 42, weakOpponent : bool = False, addtionalRewards: List[str] = None) -> gym.vector.AsyncVectorEnv:
    envs = gym.vector.AsyncVectorEnv(
        [_CreateHockyEnvWrapper(seed=seed+i,weakOpponent=weakOpponent,addtionalRewards=addtionalRewards) for i in range(envNum)]
    )
    return envs
    

def _CreateHockyEnvWrapper(seed : int = 42, weakOpponent : bool = False, addtionalRewards: List[str] = None) -> Callable:
    def _init():
        return CreateHockyEnv(seed=seed,weakOpponent=weakOpponent,addtionalRewards=addtionalRewards) 
    return _init

def CreateHockyEnv(seed : int = 42, weakOpponent : bool = False, addtionalRewards: List[str] = None) -> HockeyEnv_SB3:
    env = HockeyEnv_SB3(
        weak_opponent = weakOpponent,
        additional_rewards = addtionalRewards,
    )
    env.seed(seed)
    return env


def CreateHockyEnvSelfVector() -> gym.vector.AsyncVectorEnv:
    agentList = list(LoadTD7Agents().values())
    envs = gym.vector.AsyncVectorEnv(
        [_CreateHoeckyAgentWrapper(agent=agent) for agent in agentList]
    )
    
    return envs


def CreateHockyEnvAllVector() -> gym.vector.AsyncVectorEnv:
    td7Agents = LoadTD7Agents()
    ppoAgents = LoadAllPPOAgents()
    sacAgents = LoadAllSacAgents()
    agentList = []
    agentList.append(BasicOpponent(weak=False))
    agentList.append(BasicOpponent(weak=True))
    agentList.append(td7Agents["td7_all"])
    agentList.append(td7Agents["td7_plain"])
    agentList.append(td7Agents["td7_offensive_pressure_self"])
    agentList.append(td7Agents["td7_offensive_pressure"])
    agentList.append(sacAgents["sac_reward"])
    agentList.append(sacAgents["sac_vanilla"])
    agentList.append(sacAgents["sac_pink"])
    agentList.append(sacAgents["sac_brown"])
    agentList.append(ppoAgents["ppo_pp+op"])
    agentList.append(ppoAgents["ppo_offensive_pressure"])
    agentList.append(ppoAgents["ppo_vanilla"])
    agentList.append(ppoAgents["ppo_rnd_e1_i0.01"])

    envs = gym.vector.AsyncVectorEnv(
        [_CreateHoeckyAgentWrapper(agent=agent) for agent in agentList]
    )
    
    return envs

def _CreateHoeckyAgentWrapper(agent) -> Callable:
    def _init():
        env = HockeyEnv_SB3()
        env.opponent = agent
        return env
    return _init

def CreatePendulumV1Env() -> gym.Env:
    env = gym.make("Pendulum-v1")
    env = TimeLimit(env, max_episode_steps=500) 
    return env

def CreatePendulumV1EnvVector(envNum : int = 16) -> gym.vector.AsyncVectorEnv:
    envs = gym.vector.AsyncVectorEnv(
        [_CreatePendulumV1EnvWrapper() for _ in range(envNum)]
    )
    return envs

def _CreatePendulumV1EnvWrapper() -> Callable:
    def _init():
        return CreatePendulumV1Env() 
    return _init

def CreateHockyAgentFromTeam(algo : str = "td7", name : str = "td7_offensive_pressure") -> HockeyAgent:
    if algo == "ppo":
        return LoadAllPPOAgents()[name]
    
    if algo == "td7":
        return LoadTD7Agents()[name]
    
    if algo == "sac":
        return LoadAllSacAgents()[name]
    
    raise ValueError(f"Invalid algo: {algo}")
    

def CreateHockyEnvFromOpponent(opponent : HockeyAgent) -> HockeyEnv_SB3:
    env = HockeyEnv_SB3()
    env.opponent = opponent
    return env