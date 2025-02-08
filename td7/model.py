import os
import json
import torch
import gymnasium as gym
from .internal import Actor, Critic, Encoder, AvgL1Norm, LapHuber
from .initilizer import CreateExplorationNoise
from .replay_buffer import PER
from fvcore.common.config import CfgNode
from typing import Callable
import copy
from datetime import datetime
import csv

class TD7:
    def __init__(self, 
                 config: CfgNode, 
                 actionSpace: gym.Space, 
                 obsSpace: gym.Space
                 ):
        self.config: CfgNode = config
        self.hyperparameters: CfgNode = config.hyperparameters

        self.actionSpace: gym.Space = actionSpace
        self.obsSpace: gym.Space = obsSpace
        self.activeFunc: Callable[[torch.Tensor], torch.Tensor] = torch.relu

        self.device: torch.device = torch.device(self.hyperparameters.device)
        self.noise: Callable[[torch.Tensor], torch.Tensor] = CreateExplorationNoise(self.hyperparameters.exploration_noise)

        self.modelSaveDir : str = os.path.join(self.config.models_dir, self.config.name, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        self.trainingSteps : int = 0

        self._setPER()
        self._setNetworks()
        self._setClipValues()
        self._setCheckpointing()

    def selectAction(self, state: torch.Tensor, useCheckpoint: bool = False, useExploration : bool = True) -> torch.Tensor:
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float, device=self.device)

            if useCheckpoint:
                zs = self.checkpointEncoder.zs(state)
                action = self.checkpointActor(state, zs)
            else:
                zs = self.fixedEncoder.zs(state)
                action = self.actor(state, zs)

            action += self.noise(action)

            return action.clamp(-1, 1).cpu().numpy() * self.actionSpace.high

    def maybeTrainAndCheckpoint(self, epTimesteps: int, epReturn: float) -> None:
        self.epsSinceUpdate += 1
        self.timestepsSinceUpdate += epTimesteps
        self.minReturn = min(self.minReturn, epReturn)

        if self.minReturn < self.bestMinReturn:
            self.trainAndReset()
        elif self.epsSinceUpdate == self.maxEpsBeforeUpdate:
            self.bestMinReturn = self.minReturn
            self.checkpointActor.load_state_dict(self.actor.state_dict())
            self.checkpointEncoder.load_state_dict(self.fixedEncoder.state_dict())
            self.trainAndReset()

    def trainAndReset(self) -> None:
        for _ in range(self.timestepsSinceUpdate):
            if self.trainingSteps == self.hyperparameters.steps_before_checkpointing:
                self.bestMinReturn *= self.hyperparameters.reset_weight
                self.maxEpsBeforeUpdate = self.hyperparameters.max_eps_when_checkpointing
            
            self.train()

        self.epsSinceUpdate = 0
        self.timestepsSinceUpdate = 0
        self.minReturn = 1e8

    def train(self) -> None:
        self.trainingSteps += 1
        state, action, nextState, reward, notDone = self.replayBuffer.sample()

        with torch.no_grad():
            nextZs = self.encoder.zs(nextState)

        zs = self.encoder.zs(state)
        predZs = self.encoder.zsa(zs, action)
        encoderLoss = torch.nn.functional.mse_loss(predZs, nextZs)

        self.encoderOptimizer.zero_grad()
        encoderLoss.backward()
        self.encoderOptimizer.step()

        with torch.no_grad():
            fixedTargetZs = self.fixedEncoderTarget.zs(nextState)

            noise = (self.noise(action) * self.hyperparameters.target_policy_noise).clamp(
                -self.hyperparameters.noise_clip, self.hyperparameters.noise_clip
            )
            nextAction = (self.actorTarget(nextState, fixedTargetZs) + noise).clamp(-1, 1)
            fixedTargetZsa = self.fixedEncoderTarget.zsa(fixedTargetZs, nextAction)

            qTarget = self.criticTarget(nextState, nextAction, fixedTargetZsa, fixedTargetZs).min(1, keepdim=True)[0]
            qTarget = reward + notDone * self.hyperparameters.discount * qTarget.clamp(self.minTarget, self.maxTarget)

            self.maxValue = max(self.maxValue, float(qTarget.max()))
            self.minValue = min(self.minValue, float(qTarget.min()))

            fixedZs = self.fixedEncoder.zs(state)
            fixedZsa = self.fixedEncoder.zsa(fixedZs, action)

        q = self.critic(state, action, fixedZsa, fixedZs)
        tdLoss = (q - qTarget).abs()
        criticLoss = LapHuber(tdLoss)

        self.criticOptimizer.zero_grad()
        criticLoss.backward()
        self.criticOptimizer.step()

        priority = tdLoss.max(1)[0].clamp(min=self.hyperparameters.min_priority).pow(self.hyperparameters.alpha)
        self.replayBuffer.updatePriority(priority)

        if self.trainingSteps % self.hyperparameters.policy_freq == 0:
            actor = self.actor(state, fixedZs)
            fixedZsa = self.fixedEncoder.zsa(fixedZs, actor)
            q = self.critic(state, actor, fixedZsa, fixedZs)

            actorLoss = -q.mean()
            self.actorOptimizer.zero_grad()
            actorLoss.backward()
            self.actorOptimizer.step()
        
        if self.trainingSteps % self.hyperparameters.target_update_rate == 0:
            self.actorTarget.load_state_dict(self.actor.state_dict())
            self.criticTarget.load_state_dict(self.critic.state_dict())
            self.fixedEncoderTarget.load_state_dict(self.fixedEncoder.state_dict())
            self.fixedEncoder.load_state_dict(self.encoder.state_dict())

            self.replayBuffer.resetMaxPriority()

            self.maxTarget = self.maxValue
            self.minTarget = self.minValue

    def addEvalRes(self,timeStep : int , avgReward : float , stdReward : float) -> None:
        path = os.path.join(self.modelSaveDir, "eval.csv")
        fileExists = os.path.exists(path)

        with open(path, mode="a", new_line = '') as f:
            writer = csv.writer(f)

            if not fileExists:
                writer.writerow(["Time Step", "Average Reward", "Reward Std"])

            writer.writerow([timeStep, avgReward, stdReward])

    def saveModel(self) -> None:
        os.makedirs(self.modelSaveDir, exist_ok=True)

        torch.save({
        'actor': self.actor.state_dict(),
        'critic': self.critic.state_dict(),
        'encoder': self.encoder.state_dict(),
        'actor_optimizer': self.actorOptimizer.state_dict(),
        'critic_optimizer': self.criticOptimizer.state_dict(),
        'encoder_optimizer': self.encoderOptimizer.state_dict(),
        'training_steps': self.trainingSteps
        }, os.path.join(self.modelSaveDir, "checkpoint.pt"))

        with open(os.path.join(self.modelSaveDir, "info.json"), "w") as f:
            json.dump(self.config, f, indent=4)

        print(f"Model saved to {self.modelSaveDir}")

    def loadModel(self, path: str) -> None:
        path = os.path.join(self.config.models_dir, self.config.name, path)
        
        checkpoint = torch.load(path, map_location=self.device)

        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.encoder.load_state_dict(checkpoint['encoder'])

        self.actorOptimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.criticOptimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.encoderOptimizer.load_state_dict(checkpoint['encoder_optimizer'])

        self.trainingSteps = checkpoint.get('training_steps', 0)
        print(f"Model loaded from {path}")

    def _setPER(self) -> None:
        self.replayBuffer = PER(
            stateDim=self.obsSpace.shape[0],
            actionDim=self.actionSpace.shape[0],
            device=self.device,
            maxSize=self.hyperparameters.buffer_size,
            batchSize=self.hyperparameters.batch_size
        )

    def _setClipValues(self) -> None:
        self.maxValue : float = -1e8
        self.minValue : float = 1e8
        self.maxTarget : float = 0
        self.minTarget : float = 0

    def _setCheckpointing(self) -> None:
        self.epsSinceUpdate : int = 0
        self.timestepsSinceUpdate : int = 0
        self.maxEpsBeforeUpdate : int = 1
        self.minReturn : float = 1e8
        self.bestMinReturn : float = -1e8

        self.checkpointActor = copy.deepcopy(self.actor)
        self.checkpointEncoder = copy.deepcopy(self.encoder)

    def _setNetworks(self) -> None:
        stateDim = self.obsSpace.shape[0]
        actionDim = self.actionSpace.shape[0]
        zsDim = self.hyperparameters.zs_dim
        hDim = self.hyperparameters.h_dim
        activ = self.activeFunc
        lr = self.hyperparameters.learning_rate

        self.encoder = Encoder(stateDim = stateDim,actionDim = actionDim,zsDim = zsDim,
                               hDim= hDim, activ = activ).to(self.device)
        self.encoderOptimizer = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        self.fixedEncoder = copy.deepcopy(self.encoder)
        self.fixedEncoderTarget = copy.deepcopy(self.encoder)


        self.actor = Actor(stateDim = stateDim,actionDim = actionDim,zsDim = zsDim,
                               hDim= hDim, activ = activ).to(self.device)
        self.actorOptimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.actorTarget = copy.deepcopy(self.actor)


        self.critic = Critic(stateDim = stateDim,actionDim = actionDim,zsDim = zsDim,
                               hDim= hDim, activ = activ).to(self.device)
        self.criticOptimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.criticTarget = copy.deepcopy(self.critic)
