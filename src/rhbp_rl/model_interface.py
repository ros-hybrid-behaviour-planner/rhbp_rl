from abc import ABCMeta, abstractmethod


class AbstractDDQApproximator(object):
    '''
    Abstract approximator class that provides interface to predictive models, the implementations
    are not supposed to implement any kind of algorythmic structures, only approximative model,
    way to get a prediction for it and train it
    :param num_inputs: number of inputs for the approximator, depends on the state of the task at hand
    :param num_outputs: number of outputs, generally depends the number of actions if DDQN is used
    '''

    __metaclass__ = ABCMeta

    def __init__(self, num_inputs, num_outputs):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

    @abstractmethod
    def re_init(self, num_inputs, num_outputs):
        '''
        Reninitializes the network with the new number of inputs and outputs, generally better not to use
        '''
        pass

    @abstractmethod
    def predict(self, input_state):
        '''
        This is a basic predict method that recieves the state and returns the approximated values
        :param input_state: the observation to infer from
        :return: approximation of desired values (e.g. Q-values, state value etc)
        '''
        pass

    @abstractmethod
    def train(self, data):
        '''
        Trains the network with data
        :param data: training data in form of batch with 4 member tuple frames like [(last_state, new_state, last_action, reward)]
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
        Loads model from a checkpoint, this method should include the check on whether the checkpoint exists 
        and skip loading if it does not
        :param path: path to checkpoint file
        :return: should return true if model was loaded and false if not
        '''
        pass


    @abstractmethod
    def sync_nets(self, to_sync, tau):
        '''
        Copies the weights of to_sync model to the current modek
        :param to_sync: q network, the weight of which are used for update
        '''

    @abstractmethod
    def produce_target(self):
        '''
        This method should return a clone of the q-network made in an appropriate way
        '''