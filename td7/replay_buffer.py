import numpy as np
import torch
from torch import Tensor


class PER:
    """
    Prioritized Experience Replay (PER) buffer for reinforcement learning.

    This buffer prioritizes samples based on their importance, using a 
    cumulative sum-based sampling strategy. It maintains separate buffers 
    for states, actions, next states, rewards, and termination flags, 
    along with priorities for efficient sampling.

    Features:
    - Stores experience tuples up to a maximum buffer size.
    - Samples transitions based on priority values.
    - Updates priorities dynamically after learning.
    - Implements a priority reset mechanism.

    Args:
        stateDim (int): Dimension of the state space.
        actionDim (int): Dimension of the action space.
        device (torch.device): Device to store tensors (CPU or GPU).
        maxSize (int): Maximum number of samples to store in the buffer.
        batchSize (int): Number of samples to return per batch.
    """
    def __init__(
        self,
        stateDim: int,
        actionDim: int,
        device: torch.device,
        maxSize: int,
        batchSize: int
    ):
        """
        Initializes the PER buffer with allocated memory.

        Args:
            stateDim (int): Dimension of the state space.
            actionDim (int): Dimension of the action space.
            device (torch.device): Torch device for storing tensors.
            maxSize (int): Maximum buffer capacity.
            batchSize (int): Number of samples per training batch.
        """
        self.maxSize: int = int(maxSize)
        self.ptr: int = 0
        self.size: int = 0
        self.device: torch.device = device
        self.batchSize: int = batchSize

        self.state: np.ndarray = np.zeros((self.maxSize, stateDim), dtype=np.float32)
        self.action: np.ndarray = np.zeros((self.maxSize, actionDim), dtype=np.float32)
        self.nextState: np.ndarray = np.zeros((self.maxSize, stateDim), dtype=np.float32)
        self.reward: np.ndarray = np.zeros((self.maxSize, 1), dtype=np.float32)
        self.notDone: np.ndarray = np.zeros((self.maxSize, 1), dtype=np.float32)

        self.priority: Tensor = torch.zeros(self.maxSize, device=device)
        self.maxPriority: float = 1.0

    def add(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        nextStates: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray
    ) -> None:
        """
        Adds new experiences to the replay buffer.

        Args:
            states (np.ndarray): Batch of state observations.
            actions (np.ndarray): Batch of actions taken.
            nextStates (np.ndarray): Batch of next state observations.
            rewards (np.ndarray): Batch of received rewards.
            dones (np.ndarray): Batch of done flags (1 if episode ended, 0 otherwise).
        """
        batchSize: int = len(states)
        idx: np.ndarray = np.arange(self.ptr, self.ptr + batchSize) % self.maxSize

        self.state[idx] = states
        self.action[idx] = actions 
        self.nextState[idx] = nextStates
        self.reward[idx] = rewards
        self.notDone[idx] = 1.0 - dones

        self.priority[idx] = self.maxPriority

        self.ptr = (self.ptr + batchSize) % self.maxSize
        self.size = min(self.size + batchSize, self.maxSize)

    def sample(self) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Samples a batch of experiences from the replay buffer using prioritized sampling.

        Returns:
            tuple[Tensor, Tensor, Tensor, Tensor, Tensor]: 
                - Sampled states.
                - Sampled actions.
                - Sampled next states.
                - Sampled rewards.
                - Sampled "not done" flags.
        """
        cumsumPriority: Tensor = torch.cumsum(self.priority[:self.size], dim=0)
        randomValues: Tensor = torch.rand(size=(self.batchSize,), device=self.device) * cumsumPriority[-1]
        indices: np.ndarray = torch.searchsorted(cumsumPriority, randomValues).cpu().numpy()

        self.indices: np.ndarray = indices
        return (
            torch.tensor(self.state[indices], dtype=torch.float, device=self.device),
            torch.tensor(self.action[indices], dtype=torch.float, device=self.device),
            torch.tensor(self.nextState[indices], dtype=torch.float, device=self.device),
            torch.tensor(self.reward[indices], dtype=torch.float, device=self.device),
            torch.tensor(self.notDone[indices], dtype=torch.float, device=self.device),
        )

    def updatePriority(self, priority: Tensor) -> None:
        """
        Updates the priority values of the most recently sampled batch.

        Args:
            priority (Tensor): Updated priority values for the sampled transitions.
        """
        self.priority[self.indices] = priority.reshape(-1).detach()
        self.maxPriority = max(float(priority.max()), self.maxPriority)

    def resetMaxPriority(self) -> None:
        """
        Resets the maximum priority value based on the current buffer state.
        """
        self.maxPriority = float(self.priority[:self.size].max())
