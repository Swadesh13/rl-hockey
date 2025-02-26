import torch
import torch.nn as nn
import torch.optim as optim


class RNDModel(nn.Module):
    """
    Random Network Distillation (RND) Model.
    The target network is randomly initialized and fixed, while the predictor network is trained.
    """

    def __init__(self, input_dim, hidden_dim=256):
        """
        Initializes the RND Model.

        Args:
            input_dim (int): Dimension of input features.
            hidden_dim (int, optional): Number of hidden units. Defaults to 256.
        """
        super().__init__()

        self.target = self._get_target_network(input_dim, hidden_dim)
        self.predictor = self._get_predictor_network(input_dim, hidden_dim)

        # Freeze target network
        for param in self.target.parameters():
            param.requires_grad = False

        # optimizer has only predictor parameters
        self.optimizer = optim.Adam(self.predictor.parameters(), lr=1e-4)

        # Running mean/var states
        self.obs_mean = torch.zeros(input_dim)
        self.obs_var = torch.ones(input_dim)
        self.alpha = 0.999  # smoothing factor for mean/var updates

    def update_obs_stats(self, obs):
        """
        Updates running statistics of observations.

        Args:
            obs (torch.Tensor): Batch of observations.
        """
        # Online update of mean and variance
        batch_mean = obs.mean(dim=0)
        batch_var = obs.var(dim=0, unbiased=False)

        # exponential moving average
        self.obs_mean = self.alpha * self.obs_mean + (1 - self.alpha) * batch_mean
        self.obs_var = self.alpha * self.obs_var + (1 - self.alpha) * batch_var

    def normalize_obs(self, obs):
        """
        Normalizes observations using running statistics.

        Args:
            obs (torch.Tensor): Raw observations.

        Returns:
            torch.Tensor: Normalized and clipped observations.
        """
        # Convert to dimension [batch_size, input_dim]
        mean = self.obs_mean.unsqueeze(0)
        var = self.obs_var.unsqueeze(0)
        obs_norm = (obs - mean) / torch.sqrt(var + 1e-8)
        obs_clipped = torch.clip(obs_norm, -5.0, 5.0)
        return obs_clipped

    def forward(self, x):
        """
        Passes input through target and predictor networks.

        Args:
            x (torch.Tensor): Input data.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Predictor and target outputs.
        """
        target_output = self.target(x)
        predictor_output = self.predictor(x)
        return predictor_output, target_output

    def compute_intrinsic_reward(self, obs):
        """
        Computes intrinsic reward based on prediction error.

        Args:
            obs (np.array): Input observations.

        Returns:
            np.array: Computed intrinsic rewards.
        """
        obs = torch.tensor(obs, dtype=torch.float32)
        self.update_obs_stats(obs)
        obs = self.normalize_obs(obs)

        pred, target = self.forward(obs)
        intrinsic_reward = torch.mean((pred - target) ** 2, dim=-1).detach().numpy()
        return intrinsic_reward

    def update(self, obs):
        """
        Trains the predictor network.

        Args:
            obs (np.array): Batch of observations.

        Returns:
            float: Training loss.
        """
        obs = torch.tensor(obs, dtype=torch.float32)
        self.update_obs_stats(obs)
        obs = self.normalize_obs(obs)

        pred, target = self.forward(obs)
        loss = torch.mean((pred - target) ** 2)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def _get_target_network(self, input_dim, hidden_dim):
        """
        Creates the target network with fixed random weights.

        Args:
            input_dim (int): Input dimension.
            hidden_dim (int): Number of hidden units.

        Returns:
            nn.Sequential: Target network.
        """
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def _get_predictor_network(self, input_dim, hidden_dim):
        """
        Creates the predictor network to learn target network representations.

        Args:
            input_dim (int): Input dimension.
            hidden_dim (int): Number of hidden units.

        Returns:
            nn.Sequential: Predictor network.
        """
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
