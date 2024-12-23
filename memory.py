import numpy as np


# class to store transitions
class ReplayBuffer:
    def __init__(self, size=1e5):
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
