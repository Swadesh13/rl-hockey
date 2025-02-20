from td7.agent import TD7Agent
from typing import Dict
from henv.env import HockeyEnv_SB3

def LoadTD7Agents(evalEnv: HockeyEnv_SB3 = None) -> Dict[str, TD7Agent]:
    return {
        "td7_plain": TD7Agent(config=None, model=None, trainEnv=None, evalEnv=evalEnv, loadModel=True, modelsDir="models", modelName="td7_plain"),
        "td7_puck_proximity": TD7Agent(config=None, model=None, trainEnv=None, evalEnv=evalEnv, loadModel=True, modelsDir="models", modelName="td7_puck_proximity"),
        "td7_offensive_pressure": TD7Agent(config=None, model=None, trainEnv=None, evalEnv=evalEnv, loadModel=True, modelsDir="models", modelName="td7_offensive_pressure"),
        "td7_offensive_pressure_puck_proximity": TD7Agent(config=None, model=None, trainEnv=None, evalEnv=evalEnv, loadModel=True, modelsDir="models", modelName="td7_offensive_pressure_puck_proximity"),
    }
