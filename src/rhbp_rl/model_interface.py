from abc import ABC, abstractmethod

class AbstractApproximator(ABC):
    '''
    Abstract approximator class that provides interface to predictive models, the implementations
    are not supposed to implement any kind of algorythmic structures, only approximative model,
    way to get a prediction for it and train it
    '''
    def __init__(self):
        super(AbstractApproximator, self).__init__()

    @abstractmethod
    def predict(self, input_state):
        '''
        This is a basic predict method thatz receives recieves the state and returns the approximated values
        :param input_state: the observation to infer from
        :return: approximation of desired values (e.g. Q-values, state value etc)
        '''
        pass

    @abstractmethod
    def train(self, data):
        '''
        Trains the network with data
        :param data: training data in any shape of form, the method should now how to unpack it
        :return: nothing
        '''
        pass

    @abstractmethod
    def save_model(self):
        '''
        Saves the state of the model into a file
        :return: preferable to return string with a path to saved checkpoint
        '''
        pass
    
    @abstractmethod
    def load_model(self, path):
        '''
        Loads model from a checkpoint
        :param path: path to checkpoint file
        '''
        pass
