import numpy as np
from stable_baselines3.common.buffers import ReplayBuffer


class ExperienceMemory(ReplayBuffer):
    """
    Samples based on number of times sampled previously. Prevents most seen samples to be sampled more.
    """

    def __init__(self, size, obs_space, ac_space, *args, **kwargs):
        super().__init__(size, obs_space, ac_space, *args, **kwargs)
        self.experience = np.array([])

    def add(self, *args, **kwargs):
        super().add(*args, **kwargs)
        if len(self.experience) == self.buffer_size:
            self.experience[(self.pos - 1) % self.buffer_size] = 1
        else:
            self.experience = np.append(self.experience, 1)

    def sample(self, batch_size: int, env=None):
        experience_weight = 1 / self.experience
        upper_bound = self.buffer_size if self.full else self.pos
        batch_inds = np.random.choice(
            upper_bound,
            batch_size,
            batch_size > upper_bound,
            experience_weight / experience_weight.sum(),
        )
        self.experience[batch_inds] += 1
        return self._get_samples(batch_inds, env=env)


class PrioritizedMemory(ReplayBuffer):
    """
    Implementation of PER.
    """

    def __init__(self, size, obs_space, ac_space, alpha=0.6, *args, **kwargs):
        super().__init__(size, obs_space, ac_space, *args, **kwargs)
        self.alpha = alpha
        self.priorities = np.array([])
        self.curr_indices = None

    def add(self, *args, **kwargs):
        super().add(*args, **kwargs)
        max_priority = max(self.priorities) if len(self.priorities) else 1.0
        if len(self.priorities) == self.buffer_size:
            self.priorities[(self.pos - 1) % self.buffer_size] = max_priority
        else:
            self.priorities = np.append(self.priorities, max_priority)

    def sample(self, batch_size, env=None, beta=0.4):
        priorities = self.priorities.copy() ** self.alpha
        probs = priorities / priorities.sum()
        upper_bound = self.buffer_size if self.full else self.pos
        batch_inds = np.random.choice(
            upper_bound, batch_size, batch_size > upper_bound, probs
        )
        samples = self._get_samples(batch_inds, env=env)
        self.curr_indices = batch_inds

        # Compute importance-sampling weights
        total = upper_bound
        weights = (total * probs[batch_inds]) ** (-beta)
        weights /= weights.max()
        return samples, weights

    def update_priorities(self, priorities):
        self.priorities[self.curr_indices] = priorities


class PrioritizedExperienceMemory(ReplayBuffer):
    """
    Sample based on td-errors. Replace past samples based on experience. Removes samples that have been experienced many times already.
    """

    def __init__(self, size, obs_space, ac_space, *args, **kwargs):
        super().__init__(size, obs_space, ac_space, *args, **kwargs)
        self.experience = np.array([-1] * self.buffer_size)
        self.priorities = np.array([-1.0] * self.buffer_size)
        self.curr_indices = None

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: list,
    ) -> None:
        assert not self.optimize_memory_usage, (
            "Cannot use this class for optimized memory usage, since pos is randomly determined"
        )

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.n_envs, self.action_dim))

        # Copy to avoid modification by reference
        self.observations[self.pos] = np.array(obs)
        self.next_observations[self.pos] = np.array(next_obs)
        self.actions[self.pos] = np.array(action)
        self.rewards[self.pos] = np.array(reward)
        self.dones[self.pos] = np.array(done)

        if self.handle_timeout_termination:
            self.timeouts[self.pos] = np.array(
                [info.get("TimeLimit.truncated", False) for info in infos]
            )

        max_priority = max(self.priorities) if self.pos else 1.0
        self.priorities[self.pos] = max_priority
        self.experience[self.pos] = 1

        if self.full:
            assert -1.0 not in self.priorities
            self.pos = np.argmin(self.priorities)
        else:
            self.pos += 1
            if self.pos == self.buffer_size:
                self.full = True
                self.pos = np.argmin(self.priorities)

    def sample(self, batch_size: int, env=None):
        upper_bound = self.buffer_size if self.full else self.pos
        experience_weight = 1 / self.experience[:upper_bound]
        batch_inds = np.random.choice(
            upper_bound,
            batch_size,
            batch_size > upper_bound,
            experience_weight / experience_weight.sum(),
        )
        self.experience[batch_inds] += 1
        self.curr_indices = batch_inds
        return self._get_samples(batch_inds, env=env)

    def update_priorities(self, priorities):
        self.priorities[self.curr_indices] = priorities
