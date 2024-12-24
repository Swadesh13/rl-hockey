import numpy as np


# class to store transitions
class Memory:
    def __init__(self, size=int(1e5)):
        self.buffer = []
        self.max_size = size
        self.ptr = 0

    def add(self, transition):
        if len(self.buffer) < self.max_size:
            self.buffer.append(transition)
        else:
            self.buffer[self.ptr] = transition
        self.ptr = (self.ptr + 1) % self.max_size

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size)
        return [self.buffer[idx] for idx in indices]


class ExperienceMemory(Memory):
    def __init__(self, size=int(1e5)):
        super().__init__(size)
        self.experience = np.array([])

    def add(self, transition):
        if len(self.buffer) < self.max_size:
            self.buffer.append(transition)
            self.experience = np.append(self.experience, 1)
        else:
            self.buffer[self.ptr] = transition
            self.experience[self.ptr] = 1
        self.ptr = (self.ptr + 1) % self.max_size

    def sample(self, batch_size):
        experience_weight = 1 / self.experience
        indices = np.random.choice(len(self.buffer), batch_size, False, experience_weight / experience_weight.sum())
        self.experience[indices] += 1
        return [self.buffer[idx] for idx in indices]


class PrioritizedMemory(Memory):
    def __init__(self, size=int(1e5), alpha=0.6):
        super().__init__(size)
        self.alpha = alpha
        self.priorities = []

    def add(self, transition, priority=1.0):
        max_priority = max(self.priorities) if self.buffer else priority
        if len(self.buffer) < self.max_size:
            self.buffer.append(transition)
            self.priorities.append(max_priority)
        else:
            self.buffer[self.ptr] = transition
            self.priorities[self.ptr] = max_priority
        self.ptr = (self.ptr + 1) % self.max_size

    def sample(self, batch_size, beta=0.4):
        priorities = np.array(self.priorities) ** self.alpha
        probs = priorities / priorities.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        # Compute importance-sampling weights
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        return samples, indices, weights

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
