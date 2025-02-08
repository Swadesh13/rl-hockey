import numpy as np
import torch
from torch import Tensor


class PER:
    def __init__(
        self,
        stateDim: int,
        actionDim: int,
        device: torch.device,
        maxSize: int,
        batchSize: int
    ):
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
        self.priority[self.indices] = priority.reshape(-1).detach()
        self.maxPriority = max(float(priority.max()), self.maxPriority)

    def resetMaxPriority(self) -> None:
        self.maxPriority = float(self.priority[:self.size].max())
