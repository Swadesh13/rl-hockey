import numpy as np
import torch
from torch import nn
from torch import optim
from torch.distributions.normal import Normal
from feedforward import MLP
from noise import ColoredNoiseProcess
from norm_flows import RealNVPPolicy
from memory import PrioritizedMemory

GAMMA = 0.99
TAU = 0.005
LR = 3e-4
ALPHA = 0.2
BUFFER_SIZE = int(1e5)
BATCH_SIZE = 256


class GaussianPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, noise_color="pink", noise_seq_len=int(1e4), device="cpu"):
        super(GaussianPolicy, self).__init__()
        self.net = MLP(state_dim, hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        self.gen = ColoredNoiseProcess(color=noise_color, size=(action_dim, noise_seq_len))
        self.device = device

    def forward(self, state):
        x = self.net(state)
        mean = self.mean(x)
        log_std = self.log_std(x).clamp(-20, 2)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self(state)
        std = log_std.exp()
        cn_sample = torch.tensor(self.gen.sample()).float().to(self.device)
        dist = Normal(mean, std)
        # z = dist.rsample()  # Reparameterization trick
        z = mean + std * cn_sample
        action = torch.tanh(z)
        log_prob = (dist.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)).sum(dim=-1, keepdim=True)
        return action, log_prob


# Soft Actor-Critic Agent
class SACAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim=256,
        noise_hidden_dim=64,
        noise_color="pink",
        noise_seq_len=int(1e4),
        target_update_steps=1,
        actor_nvp=False,
        device="cpu",
    ):
        if actor_nvp:
            self.actor = RealNVPPolicy(state_dim, action_dim, 4, noise_color, noise_seq_len, device=device).to(device)
        else:
            self.actor = GaussianPolicy(
                state_dim,
                action_dim,
                noise_hidden_dim,
                noise_color=noise_color,
                noise_seq_len=noise_seq_len,
                device=device,
            ).to(device)
        self.q1 = MLP(state_dim + action_dim, 1, hidden_dim).to(device)
        self.q2 = MLP(state_dim + action_dim, 1, hidden_dim).to(device)
        self.q1_target = MLP(state_dim + action_dim, 1, hidden_dim).to(device)
        self.q2_target = MLP(state_dim + action_dim, 1, hidden_dim).to(device)

        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.q_optimizer = optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=LR)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR)
        self.log_alpha = torch.tensor([0.0], requires_grad=True, device=device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=LR)

        self.target_entropy = -float(action_dim)
        self.alpha = ALPHA
        self.device = device
        self.target_update_steps = target_update_steps
        self.update_steps = 0

    def act(self, state):
        if not isinstance(state, torch.FloatTensor):
            state = torch.FloatTensor(state).to(self.device)
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        return self.actor.sample(state)

    def update(self, memory):
        if len(memory.buffer) < BATCH_SIZE:
            return

        self.update_steps += 1

        if isinstance(memory, PrioritizedMemory):
            batch, indices, weights = memory.sample(BATCH_SIZE)
            weights = torch.tensor(weights).unsqueeze(-1).to(self.device)
        else:
            batch = memory.sample(BATCH_SIZE)
            weights = 1.0

        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.FloatTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)
            target_q1 = self.q1_target(torch.cat([next_states, next_actions], dim=1))
            target_q2 = self.q2_target(torch.cat([next_states, next_actions], dim=1))
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_probs
            target_q = rewards + (1 - dones) * GAMMA * target_q

        current_q1 = self.q1(torch.cat([states, actions], dim=1))
        current_q2 = self.q2(torch.cat([states, actions], dim=1))
        q_loss = ((current_q1 - target_q).pow(2) * weights).mean() + ((current_q2 - target_q).pow(2) * weights).mean()
        td_error = torch.abs([current_q1, current_q2][np.random.choice([0, 1], 1)[0]].detach() - target_q)

        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        actions_sampled, log_probs = self.actor.sample(states)
        q1 = self.q1(torch.cat([states, actions_sampled], dim=1))
        q2 = self.q2(torch.cat([states, actions_sampled], dim=1))
        actor_loss = ((self.alpha * log_probs - torch.min(q1, q2)) * weights).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach() * weights).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self.alpha = self.log_alpha.exp().item()

        # Update target networks
        if self.update_steps % self.target_update_steps == 0:
            with torch.no_grad():
                for param, target_param in zip(self.q1.parameters(), self.q1_target.parameters()):
                    target_param.data.mul_(1 - TAU)
                    target_param.data.add_(TAU * param.data)
                for param, target_param in zip(self.q2.parameters(), self.q2_target.parameters()):
                    target_param.data.mul_(1 - TAU)
                    target_param.data.add_(TAU * param.data)

        if isinstance(memory, PrioritizedMemory):
            memory.update_priorities(indices, td_error.cpu().numpy().flatten().tolist())
