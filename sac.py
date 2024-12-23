import numpy as np
import torch
from torch import nn
from torch import optim
from torch.distributions.normal import Normal
from feedforward import MLP
from noise import ColoredNoiseProcess
from norm_flows import RealNVPPolicy

GAMMA = 0.99
TAU = 0.005
LR = 3e-4
ALPHA = 0.2
TARGET_ENTROPY = -1.0  # Encourage exploration
BUFFER_SIZE = int(1e5)
BATCH_SIZE = 256


class Policy(nn.Module):
    def __init__(self, state_dim, action_dim, seq_len, noise_color="pink", hidden_dim=256):
        """
        seq_len = max episode steps
        """
        super(Policy, self).__init__()
        self.net = MLP(state_dim, hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        self.gen = ColoredNoiseProcess(color=noise_color, size=(action_dim, seq_len))

    def forward(self, state):
        x = self.net(state)
        mean = self.mean(x)
        log_std = self.log_std(x).clamp(-20, 2)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self(state)
        std = log_std.exp()
        cn_sample = torch.tensor(self.gen.sample()).float()
        dist = Normal(mean, std)
        # action = dist.rsample()  # Reparameterization trick
        action = mean + std * cn_sample
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        log_prob -= (2 * (np.log(2) - action - nn.functional.softplus(-2 * action))).sum(dim=1, keepdim=True)
        action = torch.tanh(action)
        return action, log_prob


# Soft Actor-Critic Agent
class SACAgent:
    def __init__(self, state_dim, action_dim, max_seq_len=500):
        self.actor = Policy(state_dim, action_dim, max_seq_len)
        # self.actor = RealNVPPolicy(state_dim, action_dim, 4)
        self.q1 = MLP(state_dim + action_dim, 1)
        self.q2 = MLP(state_dim + action_dim, 1)
        self.q1_target = MLP(state_dim + action_dim, 1)
        self.q2_target = MLP(state_dim + action_dim, 1)

        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.q_optimizer = optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=LR)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR)
        self.log_alpha = torch.tensor([0.0], requires_grad=True)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=LR)

        self.target_entropy = TARGET_ENTROPY
        self.alpha = ALPHA

    def update(self, replay_buffer):
        if len(replay_buffer.buffer) < BATCH_SIZE:
            return

        batch = replay_buffer.sample(BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.FloatTensor(np.array(states))
        actions = torch.FloatTensor(np.array(actions))
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones).unsqueeze(1)

        with torch.no_grad():
            next_actions, next_log_probs = self.actor(next_states)
            target_q1 = self.q1_target(torch.cat([next_states, next_actions], dim=1))
            target_q2 = self.q2_target(torch.cat([next_states, next_actions], dim=1))
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_probs
            target_q = rewards + (1 - dones) * GAMMA * target_q

        current_q1 = self.q1(torch.cat([states, actions], dim=1))
        current_q2 = self.q2(torch.cat([states, actions], dim=1))
        q_loss = (current_q1 - target_q).pow(2).mean() + (current_q2 - target_q).pow(2).mean()

        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        actions_sampled, log_probs = self.actor.sample(states)
        q1 = self.q1(torch.cat([states, actions_sampled], dim=1))
        q2 = self.q2(torch.cat([states, actions_sampled], dim=1))
        actor_loss = (self.alpha * log_probs - torch.min(q1, q2)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self.alpha = self.log_alpha.exp().item()

        # Update target networks
        with torch.no_grad():
            for param, target_param in zip(self.q1.parameters(), self.q1_target.parameters()):
                target_param.data.mul_(1 - TAU)
                target_param.data.add_(TAU * param.data)
            for param, target_param in zip(self.q2.parameters(), self.q2_target.parameters()):
                target_param.data.mul_(1 - TAU)
                target_param.data.add_(TAU * param.data)
