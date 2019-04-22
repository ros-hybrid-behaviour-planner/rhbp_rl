import numpy as np
import random
from collections import deque

class ExperienceBuffer(object):
    """
    the experience buffer saves all experiences. these experiences can later be revisited for training the model.
    therefore the training batches do not correlate because not the experiences that follow on after another
    are used for training
    TODO double check if this implementation is efficient and correct! DONE
    TODO CHANGING TO DEQUE MIGHT BE A GOOD IDEA DONE
    """

    def __init__(self, buffer_size=10000):
        self.buffer = []
        self.buffer_size = buffer_size
        self.counter = 0

    def reset(self):
        """
        reset the variables
        :return: 
        """
        self.buffer = []
        self.counter = 0

    def add(self, experience):
        """
        add a new experience
        :param experience: contains old state, new state, reward and action
        :return: 
        """
        self.counter += 1
        if len(self.buffer) + len(experience) >= self.buffer_size:
            self.buffer[0:(len(experience) + len(self.buffer)) - self.buffer_size] = []
        self.buffer.extend(experience)

    # get a random sample of the buffer
    def sample(self, size):
        """
        return a random selection of experiences with the specified size
        :param size: determines how large the sample should be
        :return: 
        """
        sample = np.reshape(np.array(random.sample(self.buffer, size)), [size, 4])
        return sample
