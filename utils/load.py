from td7.agent import TD7HockyAgent
from typing import Dict
from henv.env import HockeyEnv_SB3
from .parsing import get_default_td7_config
from sac.sac import SAC_HockeyAgent
from utils.parsing import convert_to_cfgnode, get_eval_env, load_config
from ppo.ppo import PPO_HockeyAgent

def LoadTD7Agents(evalEnv: HockeyEnv_SB3 = None) -> Dict[str, TD7HockyAgent]:
    config = get_default_td7_config()
    config.agent.save = False
    return {
        "td7_plain": TD7HockyAgent(config=config,evalEnv=evalEnv, loadModel=True, modelsDir="models", modelName="td7_plain"),
        "td7_puck_proximity": TD7HockyAgent(config=config,evalEnv=evalEnv, loadModel=True, modelsDir="models", modelName="td7_puck_proximity"),
        "td7_offensive_pressure": TD7HockyAgent(config=config,evalEnv=evalEnv, loadModel=True, modelsDir="models", modelName="td7_offensive_pressure"),
        "td7_offensive_pressure_puck_proximity": TD7HockyAgent(config=config,evalEnv=evalEnv, loadModel=True, modelsDir="models", modelName="td7_offensive_pressure_puck_proximity"),
        "td7_all": TD7HockyAgent(config=config,evalEnv=evalEnv, loadModel=True, modelsDir="models", modelName="td7_all"),
        "td7_offensive_pressure_self": TD7HockyAgent(config=config,evalEnv=evalEnv, loadModel=True, modelsDir="models", modelName="td7_offensive_pressure_self"),
        "td7_all_big": TD7HockyAgent(config=config,evalEnv=evalEnv, loadModel=True, modelsDir="models", modelName="td7_all_big"),
    }

def LoadAllSacAgents() -> Dict[str, SAC_HockeyAgent]:
    cfg = convert_to_cfgnode(load_config("configs/sac_hockey.yaml"))

    eval_env = get_eval_env()
    models = {}

    sac_vanilla = SAC_HockeyAgent(eval_env, config=cfg)
    sac_vanilla.load("models/sac/sac_vanilla")
    models["sac_vanilla"] = sac_vanilla

    sac_pink = SAC_HockeyAgent(eval_env, config=cfg)
    sac_pink.load("models/sac/sac_pink")
    models["sac_pink"] = sac_pink

    sac_brown = SAC_HockeyAgent(eval_env, config=cfg)
    sac_brown.load("models/sac/sac_brown")
    models["sac_brown"] = sac_brown

    sac_reward = SAC_HockeyAgent(eval_env, config=cfg)
    sac_reward.load("models/sac/sac_reward")
    models["sac_reward"] = sac_reward

    return models

def LoadAllPPOAgents() -> Dict[str, PPO_HockeyAgent]:
    models = [
        "ppo_vanilla",
        "ppo_gaussian_noise",
        "ppo_offensive_pressure",
        "ppo_pp+op",
        "ppo_puck_proximity",
        "ppo_rnd_e1_i0.01",
    ]
    agents = {}
    for model in models:
        cfg = convert_to_cfgnode(load_config(f"models/ppo/{model}.yaml"))
        eval_env = get_eval_env()
        agent = PPO_HockeyAgent(eval_env, config=cfg, eval=True)
        agent.load()
        agents[model] = agent
    return agents
