import os
import torch
import gymnasium as gym
from .internal import Actor, Critic, Encoder, LapHuber
from .replay_buffer import PER
from fvcore.common.config import CfgNode
from typing import Callable,List,Union
import copy
from datetime import datetime

class TD7:
    """
    TD7 is an actor-critic reinforcement learning algorithm that integrates:
    - Latent state representations via an encoder.
    - Prioritized Experience Replay (PER) for efficient sampling.
    - Double Q-learning (TD3-style) for stable critic updates.
    - Checkpointing and model saving/loading capabilities.

    Components:
    - Actor: Learns an optimal policy.
    - Critic: Estimates Q-values using a double network approach.
    - Encoder: Encodes states and state-action pairs into latent embeddings.

    Features:
    - Supports exploration noise strategies (Gaussian, Uniform).
    - Implements target networks for stability.
    - Uses periodic checkpointing to prevent catastrophic forgetting.
    - Applies Laplacian Huber loss for robust critic updates.
    
    Args:
        config (CfgNode): Configuration node containing hyperparameters.
        actionSpace (gym.Space): The action space of the environment.
        obsSpace (gym.Space): The observation space of the environment.
    """
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
        self.noise: Callable[[torch.Tensor], torch.Tensor] = self.createExplorationNoise()

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

        actorLoss = None
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

        return {
            "encoder_loss": encoderLoss.item(),
            "critic_loss": criticLoss.item(),
            "actor_loss": actorLoss.item() if actorLoss is not None else None
        }


    def saveModel(self, dir : str) -> None:

        path = os.path.join(dir, "model_checkpoint.pt")

        torch.save({
            'actor': self.actor.state_dict(),
            'actorOptimizer': self.actorOptimizer.state_dict(),
            'actorTarget': self.actorTarget.state_dict(),

            'critic': self.critic.state_dict(),
            'criticOptimizer': self.criticOptimizer.state_dict(),
            'criticTarget': self.criticTarget.state_dict(),

            'encoder': self.encoder.state_dict(),
            'encoderOptimizer': self.encoderOptimizer.state_dict(),
            'fixedEncoder': self.fixedEncoder.state_dict(),
            'fixedEncoderTarget': self.fixedEncoderTarget.state_dict(),

            'trainingSteps': self.trainingSteps,
            'minValue': self.minValue,
            'maxValue': self.maxValue,
            'minTarget': self.minTarget,
            'maxTarget': self.maxTarget,

            'epsSinceUpdate': self.epsSinceUpdate,
            'timestepsSinceUpdate': self.timestepsSinceUpdate,
            'maxEpsBeforeUpdate': self.maxEpsBeforeUpdate,
            'minReturn': self.minReturn,
            'bestMinReturn': self.bestMinReturn,

            'checkpointActor': self.checkpointActor.state_dict(),
            'checkpointEncoder': self.checkpointEncoder.state_dict(),
        }, path)

        print(f"Model saved to {self.modelSaveDir}")


    def loadModel(self, dir: str) -> None:
        path = os.path.join(dir, "model_checkpoint.pt")
        
        checkpoint = torch.load(path, map_location=self.device)

        self.actor.load_state_dict(checkpoint['actor'])
        self.actorTarget.load_state_dict(checkpoint['actorTarget'])
        self.actorOptimizer.load_state_dict(checkpoint['actorOptimizer'])

        self.critic.load_state_dict(checkpoint['critic'])
        self.criticTarget.load_state_dict(checkpoint['criticTarget'])
        self.criticOptimizer.load_state_dict(checkpoint['criticOptimizer'])

        self.encoder.load_state_dict(checkpoint['encoder'])
        self.encoderOptimizer.load_state_dict(checkpoint['encoderOptimizer'])
        self.fixedEncoder.load_state_dict(checkpoint['fixedEncoder'])
        self.fixedEncoderTarget.load_state_dict(checkpoint['fixedEncoderTarget'])

        self.trainingSteps = checkpoint['trainingSteps']
        self.minValue = checkpoint['minValue']
        self.maxValue = checkpoint['maxValue']
        self.minTarget = checkpoint['minTarget']
        self.maxTarget = checkpoint['maxTarget']

        self.epsSinceUpdate = checkpoint['epsSinceUpdate']
        self.timestepsSinceUpdate = checkpoint['timestepsSinceUpdate']
        self.maxEpsBeforeUpdate = checkpoint['maxEpsBeforeUpdate']
        self.minReturn = checkpoint['minReturn']
        self.bestMinReturn = checkpoint['bestMinReturn']

        self.checkpointActor.load_state_dict(checkpoint['checkpointActor'])
        self.checkpointEncoder.load_state_dict(checkpoint['checkpointEncoder'])

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

    def createExplorationNoise(self) -> Callable[[Union[torch.Tensor, List[torch.Tensor]]], Union[torch.Tensor, List[torch.Tensor]]]:
        config = self.hyperparameters.exploration_noise
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