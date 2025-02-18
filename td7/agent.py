from henv.hockey_agent import HockeyAgent
from henv.env import HockeyEnv_SB3
from fvcore.common.config import CfgNode
import gym.vector
import numpy as np
import time
import csv
from .model import TD7
import os
import json
from datetime import datetime

class TD7Agent(HockeyAgent):
    def __init__(self,config : CfgNode, model : TD7, trainEnv : gym.vector.AsyncVectorEnv ,evalEnv : HockeyEnv_SB3):
        super().__init__(HockeyEnv_SB3(), config) # HockeyEnv_SB3() is a placeholder, the actual env is passed as an argument

        self.config = config
        self.model = model
        self.trainEnv = trainEnv
        self.evalEnv = evalEnv

        self.saveDir = os.path.join(self.config.agent.save_dir, config.model.name,datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        os.makedirs(self.saveDir, exist_ok=True)


        with open(os.path.join(self.saveDir, "config.json"), "w") as f:
            json.dump(self.config, f, indent=4)

    def train(
        self,
        total_timesteps: int = None,
        log_interval: int = None,
        progress_bar: bool = False,
        callbacks : list = None,
    ):
        trainingConf = self.config.training
        numEnvs = self.trainEnv.num_envs
        useCheckpoint = self.config.model.hyperparameters.use_checkpoint
        totalTimesteps = 0
        states,_ = self.trainEnv.reset()
        episodeNum = 0
        learningStarts = self.config.model.hyperparameters.learning_starts
        epTotalRewards = np.zeros(numEnvs, dtype=np.float64)
        epTimeSteps = np.zeros(numEnvs, dtype=np.int32)
        self.training_timesteps = [] 
        encoderLoss = []
        criticLoss = [] 
        actorLoss = []
        avgStepReward = []
        path = os.path.join(self.saveDir, "train_stat.csv")
        trainStatFile = open(path, "a")
        writer = csv.writer(trainStatFile)
        writer.writerow(["reward", "encoder_loss", "critic_loss", "actor_loss"])


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

            if totalTimesteps >= learningStarts and not useCheckpoint:
                losses = self.model.train()
                encoderLoss.append(losses["encoder_loss"])
                criticLoss.append(losses["critic_loss"])
                actorLoss.append(losses["actor_loss"])
                avgStepReward.append(np.mean(rewards))

            if totalTimesteps % trainingConf.save_model_every == 0:
                self.save()

            if totalTimesteps >= trainingConf.eval_starts  and  totalTimesteps % trainingConf.eval_every == 0:
                self.evalMidTrain(timeStep=totalTimesteps, episode=episodeNum, evalEpisodeNum=trainingConf.eval_episode_num)

            states = nextStates
            totalTimesteps+=1

            epTotalRewards += rewards
            epTimeSteps += 1

            if np.any(dones):
                doneIndices = np.where(dones)[0]
                maxEpTotalRewardIdx = np.argmax(epTotalRewards[doneIndices])
                maxEpTotalReward = epTotalRewards[maxEpTotalRewardIdx]
                maxRewardTotalTimeStep = epTimeSteps[maxEpTotalRewardIdx]

                if totalTimesteps >= learningStarts and useCheckpoint:
                    self.model.maybeTrainAndCheckpoint(epTimesteps=maxRewardTotalTimeStep, epReturn=maxEpTotalReward)
                epTotalRewards[doneIndices] = 0
                epTimeSteps[doneIndices] = 0
                episodeNum += np.sum(dones)

            if totalTimesteps % trainingConf.log_interval == 0 and len(criticLoss) > 0:
                rows = zip(
                    avgStepReward, 
                    encoderLoss, 
                    criticLoss, 
                    actorLoss
                )
                writer.writerows(rows)  
                trainStatFile.flush()  
                encoderLoss.clear()
                criticLoss.clear()
                actorLoss.clear()
                avgStepReward.clear()

        trainStatFile.close()
        self.evalMidTrain(timeStep=totalTimesteps, episode=episodeNum, evalEpisodeNum=trainingConf.eval_episode_num)
        print("Training complete")
        print(f"Training took {time.time() - startTime} seconds")

    def save(self, path: str = None):
        self.model.saveModel(dir= self.saveDir)

    def load(self, path: str = None):
        self.model.loadModel(dir= os.path.join(self.config.agent.save_dir, self.config.model.name,self.config.evaluation.model_date))  

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

        path = os.path.join(self.saveDir, "mid_train_evaluations.csv")
        fileExists = os.path.exists(path)
        with open(path, mode="a") as f:
            writer = csv.writer(f)
            if not fileExists:
                writer.writerow(["Time Step", "Average Reward", "Reward Std"])
            writer.writerow([timeStep, avgReward, stdReward])
        
        print(f"Average total reward over {evalEpisodeNum} AVG episodes: {avgReward} STD: {stdReward}")
        print("--------------------------------------------")


    def evaluate(self, num_episodes = None, render_mode = None, opponent_right = None, modes = None, env = None):
        evalConf = self.config.evaluation
        from utils.evaluate import eval_agent
        return eval_agent(
            self,
            num_episodes=evalConf.ep_num,
            render_mode=evalConf.render_mode,
        )

    def predict(self, obs, deterministic=True):
        action =  self.model.selectAction(obs[np.newaxis, :], useExploration=not deterministic)[0]
        return action, None