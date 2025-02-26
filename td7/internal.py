import torch
import torch.nn as nn
from typing import Callable
from torch import Tensor


def AvgL1Norm(x: Tensor, eps: float = 1e-8) -> Tensor:
    """
    Computes the average L1-normalized version of the input tensor.
    
    This function normalizes the input tensor by its absolute mean along the last dimension,
    ensuring stability by clamping values to a minimum epsilon.

    Args:
        x (Tensor): Input tensor to be normalized.
        eps (float, optional): Small constant to prevent division by zero. Default is 1e-8.

    Returns:
        Tensor: L1-normalized tensor.
    """
    return x / x.abs().mean(-1, keepdim=True).clamp(min=eps)


def LapHuber(x: Tensor, minPriority: float = 1) -> Tensor:
    """
    Computes the Laplacian Huber loss, a robust loss function.

    This function applies the Huber loss formulation but with an additional
    priority threshold, which prevents excessive sensitivity to high-error samples.

    Args:
        x (Tensor): Input tensor, typically representing TD errors or losses.
        minPriority (float, optional): Threshold below which squared loss is applied,
                                       and beyond which linear loss is applied. Default is 1.

    Returns:
        Tensor: The computed Laplacian Huber loss.
    """
    return torch.where(x < minPriority, 0.5 * x.pow(2), minPriority * x).sum(1).mean()


class Actor(nn.Module):
    """
    Actor network for policy learning in reinforcement learning.
    
    This network maps states and latent representations (zs) to actions. 
    It consists of fully connected layers with a non-linear activation function 
    and outputs actions within the range of [-1, 1] using a tanh activation.
    """
    def __init__(self, stateDim: int, actionDim: int, zsDim: int, hDim : int, activ: Callable[[Tensor], Tensor]):
        super(Actor, self).__init__()

        self.activ = activ
        self.l0 = nn.Linear(stateDim, hDim)
        self.l1 = nn.Linear(zsDim + hDim, hDim)
        self.l2 = nn.Linear(hDim, hDim)
        self.l3 = nn.Linear(hDim, actionDim)

    def forward(self, state: Tensor, zs: Tensor) -> Tensor:
        a = AvgL1Norm(self.l0(state))
        a = torch.cat([a, zs], 1)
        a = self.activ(self.l1(a))
        a = self.activ(self.l2(a))
        return torch.tanh(self.l3(a))


class Encoder(nn.Module):
    """
    Encoder network for learning compact latent state representations.

    This model encodes states and state-action pairs into lower-dimensional 
    embeddings (zs and zsa). These embeddings serve as compact representations 
    for policy and value function learning.
    """
    def __init__(self, stateDim: int, actionDim: int, zsDim: int,hDim : int, activ: Callable[[Tensor], Tensor]):
        super(Encoder, self).__init__()

        self.activ = activ

        self.zs1 = nn.Linear(stateDim, hDim)
        self.zs2 = nn.Linear(hDim, hDim)
        self.zs3 = nn.Linear(hDim, zsDim)

        self.zsa1 = nn.Linear(zsDim + actionDim, hDim)
        self.zsa2 = nn.Linear(hDim, hDim)
        self.zsa3 = nn.Linear(hDim, zsDim)

    def zs(self, state):
        zs = self.activ(self.zs1(state))
        zs = self.activ(self.zs2(zs))
        zs = AvgL1Norm(self.zs3(zs))
        return zs

    def zsa(self, zs, action):
        zsa = self.activ(self.zsa1(torch.cat([zs, action], 1)))
        zsa = self.activ(self.zsa2(zsa))
        zsa = self.zsa3(zsa)
        return zsa



class Critic(nn.Module):
    """
    Critic network for Q-value estimation using a Double Q-learning approach.

    This model takes as input the state, action, and latent embeddings (zsa, zs),
    and estimates Q-values for two separate Q-networks to reduce overestimation bias.
    It follows the TD3 architecture to improve stability in policy updates.
    """
    def __init__(self, stateDim: int, actionDim: int, zsDim: int, hDim: int, activ: Callable[[Tensor], Tensor]):
        super(Critic, self).__init__()

        self.activ = activ

        self.q01 = nn.Linear(stateDim + actionDim, hDim)
        self.q1 = nn.Linear(2*zsDim + hDim, hDim)
        self.q2 = nn.Linear(hDim, hDim)
        self.q3 = nn.Linear(hDim, 1)

        self.q02 = nn.Linear(stateDim + actionDim, hDim)
        self.q4 = nn.Linear(2*zsDim + hDim, hDim)
        self.q5 = nn.Linear(hDim, hDim)
        self.q6 = nn.Linear(hDim, 1)


    def forward(self, state, action, zsa, zs):
        sa = torch.cat([state, action], 1)
        embeddings = torch.cat([zsa, zs], 1)

        q1 = AvgL1Norm(self.q01(sa))
        q1 = torch.cat([q1, embeddings], 1)
        q1 = self.activ(self.q1(q1))
        q1 = self.activ(self.q2(q1))
        q1 = self.q3(q1)

        q2 = AvgL1Norm(self.q02(sa))
        q2 = torch.cat([q2, embeddings], 1)
        q2 = self.activ(self.q4(q2))
        q2 = self.activ(self.q5(q2))
        q2 = self.q6(q2)
        return torch.cat([q1, q2], 1)

