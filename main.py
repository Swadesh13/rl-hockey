import pickle
import os
from henv.env import HockeyEnv_SB3
from henv.env import BasicOpponent

pickName = "33e58cf0-0b2a-4a30-a83a-f4f5f31ce033.pkl"


class Temp(BasicOpponent):
    def __init__(self, weak=False, keep_mode=True,actions : list = None):
        super().__init__(weak, keep_mode)
        self.actions = actions

    def getAction(self):
        for a in self.actions:
            yield a

    def predict(self, state, deterministic=None):
        return next(self.getAction()), None

def getFromPickle(path : str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, "rb") as f:
        data = pickle.load(f)
        env = HockeyEnv_SB3()
        roundNum = data["num_rounds"][0][0]
        for i in range(roundNum):
            i+=1
            obsKey = f"observations_round_{i}"
            actionKey = f"actions_round_{i}"
            epNum = len(data[obsKey])
            leftPlayerActions = [action[:4] for action in data[actionKey]]
            rightPlayerActions = [action[4:] for action in data[actionKey]]
            rightPlayer = Temp(actions=rightPlayerActions)
            env.opponent = rightPlayer
            for j in range(epNum):
                print(j)
                obs = data[obsKey][j]
                env.step(leftPlayerActions[j])
                env.render()
                

        

getFromPickle(pickName)