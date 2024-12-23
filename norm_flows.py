import torch
from torch import nn
from torch.distributions import Normal
from feedforward import MLP


class RealNVPFlow(nn.Module):
    def __init__(self, input_dim):
        super(RealNVPFlow, self).__init__()
        self.s = MLP(input_dim // 2, input_dim // 2)
        self.t = MLP(input_dim // 2, input_dim // 2)

    def forward(self, x):
        print(x)
        print(torch.chunk(x, 2, dim=1))
        x1, x2 = torch.chunk(x, 2, dim=1)
        s = self.s(x1)
        t = self.t(x1)
        z2 = x2 * torch.exp(s) + t
        z = torch.cat([x1, z2], dim=1)
        log_det_jacobian = s.sum(dim=1, keepdim=True)
        return z, log_det_jacobian

    def inverse(self, z):
        z1, z2 = torch.chunk(z, 2, dim=1)
        s = self.s(z1)
        t = self.t(z1)
        x2 = (z2 - t) * torch.exp(-s)
        x = torch.cat([z1, x2], dim=1)
        return x


class RealNVPPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, flow_steps):
        super(RealNVPPolicy, self).__init__()
        self.base_mean = MLP(state_dim, action_dim)
        self.base_log_std = MLP(state_dim, action_dim)
        self.flows = nn.ModuleList([RealNVPFlow(action_dim) for _ in range(flow_steps)])

    def forward(self, state, num_samples=3):
        mean = self.base_mean(state)
        log_std = self.base_log_std(state).clamp(-20, 2)
        std = log_std.exp()
        print("Mean", mean, std)
        base_dist = Normal(mean, std)
        z = base_dist.rsample()
        log_prob = base_dist.log_prob(z).sum(dim=-1, keepdim=True)

        for flow in self.flows:
            z, flow_log_prob = flow(z)
            log_prob += flow_log_prob

        action = torch.tanh(z)
        return action, log_prob
