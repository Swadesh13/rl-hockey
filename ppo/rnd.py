import torch
import torch.nn as nn
import torch.optim as optim


class RNDModel(nn.Module):
    """
    Random Network Distillation (RND) Model.
    The target network is randomly initialized and fixed, while the predictor network is trained.
    """

    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()

        self.target = self._get_target_network(input_dim, hidden_dim)
        self.predictor = self._get_predictor_network(input_dim, hidden_dim)

        # Freeze target network
        for param in self.target.parameters():
            param.requires_grad = False

        self.optimizer = optim.Adam(self.predictor.parameters(), lr=1e-4)

    def forward(self, x):
        target_output = self.target(x)
        predictor_output = self.predictor(x)
        return predictor_output, target_output

    def compute_intrinsic_reward(self, obs):
        """
        Compute intrinsic reward based on prediction error.
        """
        obs = torch.tensor(obs, dtype=torch.float32)
        pred, target = self.forward(obs)
        intrinsic_reward = torch.mean((pred - target) ** 2, dim=-1).detach().numpy()
        return intrinsic_reward

    def update(self, obs):
        """
        Train the predictor network.
        """
        obs = torch.tensor(obs, dtype=torch.float32)
        pred, target = self.forward(obs)
        loss = torch.mean((pred - target) ** 2)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def _get_target_network(self, input_dim, hidden_dim):
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def _get_predictor_network(self, input_dim, hidden_dim):
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
