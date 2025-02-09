import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Callable, Union
from fvcore.common.config import CfgNode
from torch import Tensor
import gymnasium as gym
from henv.env import HockeyEnv_SB3


def AvgL1Norm(x: Tensor, eps: float = 1e-8) -> Tensor:
    return x / x.abs().mean(-1, keepdim=True).clamp(min=eps)


def LapHuber(x: Tensor, minPriority: float = 1) -> Tensor:
    return torch.where(x < minPriority, 0.5 * x.pow(2), minPriority * x).sum(1).mean()


class Actor(nn.Module):
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

