import numpy as np
from stable_baselines3.common.buffers import ReplayBuffer


class ExperienceMemory(ReplayBuffer):
    def __init__(self, size, obs_space, ac_space, *args, **kwargs):
        super().__init__(size, obs_space, ac_space, *args, **kwargs)
        self.experience = np.array([])

    def add(self, *args, **kwargs):
        super().add(*args, **kwargs)
        if self.full:
            self.experience[(self.pos - 1) % self.buffer_size] = 1
        else:
            self.experience = np.append(self.experience, 1)

    def sample(self, batch_size: int, env=None):
        experience_weight = 1 / self.experience
        upper_bound = self.buffer_size if self.full else self.pos
        batch_inds = np.random.choice(upper_bound, batch_size, batch_size > upper_bound, experience_weight / experience_weight.sum())
        self.experience[batch_inds] += 1
        return self._get_samples(batch_inds, env=env)


class PrioritizedMemory(ReplayBuffer):
    def __init__(self, size, obs_space, ac_space, alpha=0.6, *args, **kwargs):
        super().__init__(size, obs_space, ac_space, *args, **kwargs)
        self.alpha = alpha
        self.priorities = []

    def add(self, *args, **kwargs):
        super().add(*args, **kwargs)
        max_priority = max(self.priorities) if len(self.priorities) else 1.0
        if self.full:
            self.priorities[(self.pos - 1) % self.buffer_size] = max_priority
        else:
            self.priorities.append(max_priority)

    def sample(self, batch_size, env=None, beta=0.4):
        priorities = np.array(self.priorities) ** self.alpha
        probs = priorities / priorities.sum()
        upper_bound = self.buffer_size if self.full else self.pos
        batch_inds = np.random.choice(upper_bound, batch_size, batch_size > upper_bound, probs)
        samples = self._get_samples(batch_inds, env=env)

        # Compute importance-sampling weights
        # total = upper_bound
        # weights = (total * probs[indices]) ** (-beta)
        # weights /= weights.max()
        return samples, batch_inds  # , weights

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = np.array(priority)[0]
