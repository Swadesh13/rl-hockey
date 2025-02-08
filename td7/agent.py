from henv.hockey_agent import HockeyAgent
from henv.env import HockeyEnv_SB3
from fvcore.common.config import CfgNode
from typing import Union,List,Callable
import gym.vector
import numpy as np
import time
import torch
from .model import TD7

class TD7Agent(HockeyAgent):
    def __init__(self,config : CfgNode, model : TD7, trainEnv : gym.vector.AsyncVectorEnv ,evalEnv : HockeyEnv_SB3):
        super().__init__(HockeyEnv_SB3(), config) # HockeyEnv_SB3() is a placeholder, the actual env is passed as an argument

        self.config = config
        self.model = model
        self.trainEnv = trainEnv
        self.evalEnv = evalEnv

    def train(
        self,
        total_timesteps: int = None,
        log_interval: int = None,
        progress_bar: bool = False,
        callbacks : list = None,
    ):
        trainingConf = self.config.training
        numEnvs = self.trainEnv.num_envs
        totalTimesteps = 0
        states,_ = self.trainEnv.reset()
        episodeNum = 0
        learningStarts = self.config.model.hyperparameters.learning_starts

        startTime = time.time()
        print("Starting training...")

        while totalTimesteps < trainingConf.total_timesteps:
            if totalTimesteps % trainingConf.log_interval == 0:
                print(f"Time step: {totalTimesteps}, Episode: {episodeNum}")


            if totalTimesteps >= learningStarts:
                actions = self.model.selectAction(states)
            else:
                actions = [self.trainEnv.single_action_space.sample() for _ in range(numEnvs)]


            nextStates, rewards, dones, _, _ = self.trainEnv.step(actions)

            self.model.replayBuffer.add(states, actions, nextStates, rewards.reshape(-1,1), dones.reshape(-1,1))

            if np.any(dones):
                episodeNum += np.sum(dones)

            if totalTimesteps >= learningStarts:
                self.model.train()

            if totalTimesteps % trainingConf.save_model_every == 0:
                self.save()

            if totalTimesteps >= trainingConf.eval_starts  and  totalTimesteps % trainingConf.eval_every == 0:
                self.evalMidTrain(timeStep=totalTimesteps, episode=episodeNum, evalEpisodeNum=trainingConf.eval_episode_num)

            states = nextStates
            totalTimesteps+=1
        
        print("Training complete")
        print(f"Training took {time.time() - startTime} seconds")

    def save(self, path: str = None):
        self.model.saveModel()

    def load(self, path: str = None):
        self.model.loadModel(path= path)

    def evalMidTrain(self, timeStep : int, episode : int, evalEpisodeNum : int):
        print(f" ---Evaluation at Time step: {timeStep}, Episode: {episode} ---")
        totalRewards = []
        for _ in range(evalEpisodeNum):
            state, _ = self.evalEnv.reset()  
            episodeReward = 0
            done = False

            while not done:
                action = self.model.selectAction(state[np.newaxis, :], useExploration=False)[0]
                next_state, reward, done, _, _ = self.evalEnv.step(action)

                episodeReward += reward
                state = next_state
                

            totalRewards.append(episodeReward)
        avgReward = np.mean(totalRewards)
        stdReward = np.std(totalRewards)
        print(f"Average total reward over {evalEpisodeNum} AVG episodes: {avgReward} STD: {stdReward}")
        self.model.addEvalRes(timeStep= timeStep, avgReward= avgReward, stdReward= stdReward)
        print("--------------------------------------------")


    def evaluate(self, num_episodes = None, render_mode = None, opponent_right = None, modes = None):
        print("--------------------------------------------")
        totalRewards = []
        for _ in range(self.config.evaluation.ep_num):
            state, _ = self.evalEnv.reset()  
            episodeReward = 0
            done = False

            while not done:
                if render_mode:
                    self.evalEnv.render(mode=render_mode)

                action = self.model.selectAction(state[np.newaxis, :], useExploration=False)[0]
                next_state, reward, done, _, _ = self.evalEnv.step(action)

                episodeReward += reward
                state = next_state
                

            totalRewards.append(episodeReward)
        avgReward = np.mean(totalRewards)
        stdReward = np.std(totalRewards)
        print(f"Average total reward over {num_episodes} AVG episodes: {avgReward} STD: {stdReward}")
        print("--------------------------------------------")

    def predict(self, obs, deterministic=True):
        return super().predict(obs, deterministic)