import numpy as np
import random
from collections import deque

class ExperienceBuffer(object):
    """
    the experience buffer saves all experiences. these experiences can later be revisited for training the model.
    therefore the training batches do not correlate because not the experiences that follow on after another
    are used for training
    """

    def __init__(self, buffer_size=150, timeseries=False):
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen = buffer_size)
        self.timeseries = timeseries


    def reset(self, timeseries=False):
        """
        reset the variables
        :return: 
        """
        self.buffer = deque(maxlen = self.buffer_size)
        self.timeseries = timeseries

    def add(self, experience):
        """
        add a new experience
        :param experience: contains old state, new state, reward and action or the whole episode sequence 
        :return: 
        """
        if self.timeseries:    
            self.buffer.append(experience)
        else:
            self.buffer.extend(experience)

        

    # get a random sample of the buffer
    def sample(self, size, length=1):
        """
        return a random selection of experiences with the specified size
        :param size: determines how large the sample should be
        :return: 
        """
        if self.timeseries:
            if len(self.buffer) < 1:
                return []
            sampled_episodes = random.sample(self.buffer,size)
            sampledTraces = []
            for episode in sampled_episodes:
                if len(episode) < length:
                    return []
                point = np.random.randint(0,len(episode)+1-length)
                sampledTraces.append(episode[point:point+length])
            sampledTraces = np.array(sampledTraces)
            return np.reshape(sampledTraces,[size*length,4])
        else:    
            if size > len(self.buffer):
                size = len(self.buffer)
            sample = np.reshape(np.array(random.sample(self.buffer, size)), [size, 4])
            return sample

    def get_length(self):
        if self.timeseries:
            return sum([len(ep) for ep in self.buffer])
        return len(self.buffer)

    def get_full_memory(self):
        return np.reshape(np.array(self.buffer, [len(self.buffer), 4]))