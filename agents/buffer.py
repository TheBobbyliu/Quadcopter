import random
from collections import namedtuple, deque

class Buffer():
    """
    Parameters:
        len: max size of the memory
        batchsize: the number of experiences to get everytime
    """
    def __init__(self, len, batch_size):
        self.memory = deque(maxlen = len)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
    def sample(self):
        return random.sample(self.memory, k = self.batch_size)
    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    def __len__(self):
        return len(self.memory)