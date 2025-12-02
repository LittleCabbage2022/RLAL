# 文件路径: src/utils/buffer.py
import random
import numpy as np

class ReplayBuffer:
    """
    Simple replay buffer that stores transitions with epoch index (et).
    Each entry: (s, a, r, s2, done, et)
    sample(batch_size) => returns arrays (s,a,r,s2,d,et)
    """
    def __init__(self, capacity):
        self.capacity = int(capacity)
        self.buffer = []
        self.pos = 0

    def push(self, state, action, reward, next_state, done, epoch_idx):
        data = (state, action, reward, next_state, done, epoch_idx)
        if len(self.buffer) < self.capacity:
            self.buffer.append(data)
        else:
            self.buffer[self.pos] = data
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s,a,r,s2,d,et = zip(*batch)
        return np.stack(s), np.stack(a), np.array(r), np.stack(s2), np.array(d), np.array(et)

    def __len__(self):
        return len(self.buffer)
