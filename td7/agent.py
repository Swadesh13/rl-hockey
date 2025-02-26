from henv.hockey_agent import HockeyAgent
from henv.env import HockeyEnv_SB3
from fvcore.common.config import CfgNode
import gymnasium as gym
import numpy as np
import time
from .model import TD7
import os
import json
from datetime import datetime
from utils.evaluate import eval_agent
from torch.utils.tensorboard import SummaryWriter
from utils.parsing import get_default_td7_config

class TD7HockyAgent(HockeyAgent):
    def __init__(self,config : CfgNode = None, model : TD7 = None , trainEnv : gym.vector.AsyncVectorEnv = None ,evalEnv : HockeyEnv_SB3 = None, loadModel : bool = None, modelsDir : str = None ,modelName : str = None):
        if not loadModel:
            self.config = config
            self.model = model
            self.trainEnv = trainEnv
            self.evalEnv = evalEnv
            self.saveDir = os.path.join(self.config.agent.save_dir, config.model.name,f"{self.config.train_env.env_name}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}")
            self.modelPath = self.saveDir
            self.writer = SummaryWriter(log_dir=self.saveDir)
            if self.config.agent.save:
                os.makedirs(self.saveDir, exist_ok=True)
                with open(os.path.join(self.saveDir, "config.json"), "w") as f:
                    json.dump(self.config, f, indent=4)
        else:
            self.modelPath = os.path.join(modelsDir,"td7",modelName)
            if not config:
                configPath = os.path.join(self.modelPath, "config.json")
                if os.path.exists(configPath):
                    with open(configPath, "r") as f:
                        config_dict = json.load(f)
                    self.config = CfgNode(config_dict)
                else:
                    config = get_default_td7_config()
                    self.config = config
            else:
                self.config = config
            if self.config.agent.save:
                self.saveDir = os.path.join(modelsDir,"td7",f"{modelName}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}")
                self.writer = SummaryWriter(log_dir=self.saveDir)
            else:
                self.writer = None
            self.evalEnv = evalEnv
            self.trainEnv = trainEnv
            self.load(path=self.modelPath)

    def train(
        self, *args, **kwargs
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

                if self.writer is not None:
                    self.writer.add_scalar("Loss/Encoder", losses["encoder_loss"] or 0, totalTimesteps)
                    self.writer.add_scalar("Loss/Critic", losses["critic_loss"] or 0, totalTimesteps)
                    self.writer.add_scalar("Loss/Actor", losses["actor_loss"] or 0, totalTimesteps)

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

        self.evalMidTrain(timeStep=totalTimesteps, episode=episodeNum, evalEpisodeNum=trainingConf.eval_episode_num)
        if self.writer is not None:
            self.writer.close()
        print("Training complete")
        print(f"Training took {time.time() - startTime} seconds")

    def save(self, path: str = None):
        self.model.saveModel(dir= self.saveDir)

    def load(self, path: str = None):
        self.model = TD7(
            config = self.config.model,
            actionSpace = self.evalEnv.action_space,
            obsSpace = self.evalEnv.observation_space
        )
        dir = path or os.path.join(self.config.agent.save_dir, self.config.model.name,self.config.model.model_load_name)
        self.model.loadModel(dir=dir)  

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

        if self.writer is not None:
            self.writer.add_scalar("Evaluation/Average Reward", avgReward, timeStep)
            self.writer.add_scalar("Evaluation/Reward Std Dev", stdReward, timeStep)
        
        print(f"Average total reward over {evalEpisodeNum} AVG episodes: {avgReward} STD: {stdReward}")
        print("--------------------------------------------")

    def evaluate(self, num_episodes = None, render_mode = None, opponent_right = None, modes = None, env = None):
        evalConf = self.config.evaluation
        res =  eval_agent(player_left= self, num_episodes=evalConf.ep_num, render_mode=evalConf.render_mode, opponent_right= opponent_right,
        )
        if self.config.agent.save:
            path = os.path.join(self.modelPath, "evaluation.txt")
            with open(path, mode="a") as f:
                f.write(str(res) + "\n")
        return res
        


    def predict(self, obs, deterministic=True):
        action =  self.model.selectAction(obs[np.newaxis, :], useExploration=not deterministic)[0]
        return action, None
    
    def act(self, obs, deterministic=True):
        action, _ =  self.predict(obs, deterministic=deterministic)
        return action
    

class TD7GymAgent(TD7HockyAgent):
    def __init__(self, config, model, trainEnv, evalEnv, loadModel = False, modelsDir = None, modelName = None):
        super().__init__(config, model, trainEnv, evalEnv, loadModel, modelsDir, modelName)

    def evaluate(self, num_episodes=None, render_mode=None, opponent_right=None, modes=None, env=None):
        print(f" ---Evauation---")
        evalConf = self.config.evaluation
        totalRewards = []
        for _ in range(evalConf.ep_num):
            state, _ = self.evalEnv.reset()  
            episodeReward = 0
            truncrated = False
            while not truncrated:
                action = self.model.selectAction(state[np.newaxis, :], useExploration=False)[0]
                next_state, reward, _, truncrated, _ = self.evalEnv.step(action)

                episodeReward += reward
                state = next_state
                
            totalRewards.append(episodeReward)
        avgReward = np.mean(totalRewards)
        stdReward = np.std(totalRewards)
        print(f"Average total reward over {evalConf.ep_num} episodes: {avgReward} STD: {stdReward}")
        print("--------------------------------------------")
        if self.config.agent.save:
            path = os.path.join(self.saveDir, "evaluation.txt")
            with open(path, mode="a") as f:
                f.write(f"AvgReward : {avgReward}, StdReward : {stdReward}" + "\n")
        
    def evalMidTrain(self, timeStep : int, episode : int, evalEpisodeNum : int):
        print(f" ---Evaluation at Time step: {timeStep}, Episode: {episode} ---")
        totalRewards = []
        for _ in range(evalEpisodeNum):
            state, _ = self.evalEnv.reset()  
            episodeReward = 0
            truncrated = False
            while not truncrated:
                action = self.model.selectAction(state[np.newaxis, :], useExploration=False)[0]
                next_state, reward, _, truncrated, _ = self.evalEnv.step(action)

                episodeReward += reward
                state = next_state
                
            totalRewards.append(episodeReward)
        avgReward = np.mean(totalRewards)
        stdReward = np.std(totalRewards)
        if self.writer is not None:
            self.writer.add_scalar("Evaluation/Average Reward", avgReward, timeStep)
            self.writer.add_scalar("Evaluation/Reward Std Dev", stdReward, timeStep)
        print(f"Average total reward over {evalEpisodeNum} AVG episodes: {avgReward} STD: {stdReward}")
        print("--------------------------------------------")


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


        startTime = time.time()
        print("Starting training...")

        while totalTimesteps < trainingConf.total_timesteps:
            if totalTimesteps % trainingConf.log_interval == 0:
                print(f"Time step: {totalTimesteps}, Episode: {episodeNum}")


            if totalTimesteps >= learningStarts:
                actions = self.model.selectAction(states)
            else:
                actions = [self.trainEnv.single_action_space.sample() for _ in range(numEnvs)]


            nextStates, rewards, dones, truncrated, _ = self.trainEnv.step(actions)

            self.model.replayBuffer.add(states, actions, nextStates, rewards.reshape(-1,1), dones.reshape(-1,1))

            if totalTimesteps >= learningStarts and not useCheckpoint:
                losses = self.model.train()

                if self.writer is not None:
                    self.writer.add_scalar("Loss/Encoder", losses["encoder_loss"] or 0, totalTimesteps)
                    self.writer.add_scalar("Loss/Critic", losses["critic_loss"] or 0, totalTimesteps)
                    self.writer.add_scalar("Loss/Actor", losses["actor_loss"] or 0, totalTimesteps)
                    self.writer.add_scalar("Reward/Average Step Reward", np.mean(rewards), totalTimesteps)
        
            if totalTimesteps % trainingConf.save_model_every == 0:
                self.save()

            if totalTimesteps >= trainingConf.eval_starts  and  totalTimesteps % trainingConf.eval_every == 0:
                self.evalMidTrain(timeStep=totalTimesteps, episode=episodeNum, evalEpisodeNum=trainingConf.eval_episode_num)

            states = nextStates
            totalTimesteps+=1

            epTotalRewards += rewards
            epTimeSteps += 1

            if np.any(truncrated):
                doneIndices = np.where(truncrated)[0]
                maxEpTotalRewardIdx = np.argmax(epTotalRewards[doneIndices])
                maxEpTotalReward = epTotalRewards[maxEpTotalRewardIdx]
                maxRewardTotalTimeStep = epTimeSteps[maxEpTotalRewardIdx]

                if totalTimesteps >= learningStarts and useCheckpoint:
                    self.model.maybeTrainAndCheckpoint(epTimesteps=maxRewardTotalTimeStep, epReturn=maxEpTotalReward)
                epTotalRewards[doneIndices] = 0
                epTimeSteps[doneIndices] = 0
                episodeNum += np.sum(truncrated)

        self.evalMidTrain(timeStep=totalTimesteps, episode=episodeNum, evalEpisodeNum=trainingConf.eval_episode_num)
        if self.writer is not None:
            self.writer.close()
        print("Training complete")
        print(f"Training took {time.time() - startTime} seconds")
