from typing import List, Callable
import gymnasium as gym
from henv.env import HockeyEnv_SB3
from henv.hockey_agent import HockeyAgent
from utils.load import LoadTD7Agents, LoadAllPPOAgents, LoadAllSacAgents
from henv.env import BasicOpponent

def CreateHockyEnvVector(envNum: int = 16, seed: int = 42, weakOpponent: bool = False, addtionalRewards: List[str] = None) -> gym.vector.AsyncVectorEnv:
    """
    Creates a vectorized Hockey environment using AsyncVectorEnv.
    
    Args:
        envNum (int): Number of parallel environments to create.
        seed (int): Random seed for environment initialization.
        weakOpponent (bool): Whether to use a weak opponent.
        addtionalRewards (List[str]): List of additional rewards to use.

    Returns:
        gym.vector.AsyncVectorEnv: A vectorized environment for parallel execution.
    """
    envs = gym.vector.AsyncVectorEnv(
        [_CreateHockyEnvWrapper(seed=seed + i, weakOpponent=weakOpponent, addtionalRewards=addtionalRewards) for i in range(envNum)]
    )
    return envs
    

def _CreateHockyEnvWrapper(seed: int = 42, weakOpponent: bool = False, addtionalRewards: List[str] = None) -> Callable:
    """
    Creates a wrapper function to initialize HockeyEnv_SB3 with specific settings.

    Args:
        seed (int): Random seed for environment initialization.
        weakOpponent (bool): Whether to use a weak opponent.
        addtionalRewards (List[str]): List of additional rewards.

    Returns:
        Callable: A function that initializes the environment when called.
    """
    def _init():
        return CreateHockyEnv(seed=seed, weakOpponent=weakOpponent, addtionalRewards=addtionalRewards) 
    return _init

def CreateHockyEnv(seed: int = 42, weakOpponent: bool = False, addtionalRewards: List[str] = None) -> HockeyEnv_SB3:
    """
    Creates a single instance of the HockeyEnv_SB3 environment.

    Args:
        seed (int): Random seed for environment initialization.
        weakOpponent (bool): Whether to use a weak opponent.
        addtionalRewards (List[str]): List of additional rewards.

    Returns:
        HockeyEnv_SB3: An instance of the Hockey environment.
    """
    env = HockeyEnv_SB3(
        weak_opponent=weakOpponent,
        additional_rewards=addtionalRewards,
    )
    env.seed(seed)
    return env


def CreateHockyEnvSelfVector() -> gym.vector.AsyncVectorEnv:
    """
    Creates a vectorized environment where agents compete against themselves.

    Returns:
        gym.vector.AsyncVectorEnv: A vectorized Hockey environment.
    """
    agentList = list(LoadTD7Agents().values())
    envs = gym.vector.AsyncVectorEnv(
        [_CreateHoeckyAgentWrapper(agent=agent) for agent in agentList]
    )
    return envs


def CreateHockyEnvAllVector() -> gym.vector.AsyncVectorEnv:
    """
    Creates a vectorized environment with all available agents, including:
    - TD7-trained agents
    - PPO-trained agents
    - SAC-trained agents
    - Basic weak and strong opponents

    Returns:
        gym.vector.AsyncVectorEnv: A vectorized Hockey environment with mixed opponents.
    """
    td7Agents = LoadTD7Agents()
    ppoAgents = LoadAllPPOAgents()
    sacAgents = LoadAllSacAgents()
    agentList = list(td7Agents.values()) + list(ppoAgents.values()) + list(sacAgents.values()) + [BasicOpponent(weak=True)] + [BasicOpponent(weak=False)]

    envs = gym.vector.AsyncVectorEnv(
        [_CreateHoeckyAgentWrapper(agent=agent) for agent in agentList]
    )
    return envs

def _CreateHoeckyAgentWrapper(agent) -> Callable:
    """
    Creates a wrapper function for initializing a Hockey environment with a specific opponent.

    Args:
        agent (HockeyAgent): The opponent agent.

    Returns:
        Callable: A function that initializes the environment with the specified opponent.
    """
    def _init():
        env = HockeyEnv_SB3()
        env.opponent = agent
        return env
    return _init

def CreateGymEnv(envName: str = "HalfCheetah-v5") -> gym.Env:
    """
    Creates a Gym environment based on the specified environment name.

    Args:
        envName (str): The name of the Gym environment to create.

    Returns:
        gym.Env: An instance of the specified Gym environment.
    """
    env = gym.make(envName)
    return env

def CreatePendulumV1EnvVector(envName: str = "HalfCheetah-v5", envNum: int = 16) -> gym.vector.AsyncVectorEnv:
    """
    Creates a vectorized Gym environment for parallel execution.

    Args:
        envName (str): The name of the Gym environment.
        envNum (int): Number of parallel environments.

    Returns:
        gym.vector.AsyncVectorEnv: A vectorized Gym environment.
    """
    envs = gym.vector.AsyncVectorEnv(
        [_CreateGymEnvWrapper(envName) for _ in range(envNum)]
    )
    return envs

def _CreateGymEnvWrapper(envName: str = "HalfCheetah-v5") -> Callable:
    """
    Creates a wrapper function for initializing a Gym environment.

    Args:
        envName (str): The name of the Gym environment.

    Returns:
        Callable: A function that initializes the environment when called.
    """
    def _init():
        return CreateGymEnv(envName) 
    return _init

def CreateHockyAgentFromTeam(algo: str = "td7", name: str = "td7_offensive_pressure") -> HockeyAgent:
    """
    Loads a HockeyAgent based on the algorithm and team name.

    Args:
        algo (str): The algorithm used for training ("td7", "ppo", or "sac").
        name (str): The agent's team name.

    Returns:
        HockeyAgent: The requested agent.

    Raises:
        ValueError: If the specified algorithm is invalid.
    """
    if algo == "ppo":
        return LoadAllPPOAgents()[name]
    
    if algo == "td7":
        return LoadTD7Agents()[name]
    
    if algo == "sac":
        return LoadAllSacAgents()[name]
    
    raise ValueError(f"Invalid algo: {algo}")
    

def CreateHockyEnvFromOpponent(opponent: HockeyAgent) -> HockeyEnv_SB3:
    """
    Creates a Hockey environment with a specific opponent.

    Args:
        opponent (HockeyAgent): The opponent agent.

    Returns:
        HockeyEnv_SB3: An instance of the Hockey environment with the specified opponent.
    """
    env = HockeyEnv_SB3()
    env.opponent = opponent
    return env
