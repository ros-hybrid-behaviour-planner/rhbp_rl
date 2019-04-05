class Database():
    '''
    Database class to provide storage for examples for training.
    Expirience buffer that will be also a service potentially
    '''
    def __init__(self):
        pass


    def get_pop_example(self):
        '''
        Returns an example
        :return: a training example (complete example that is neeeded to train the network)
        '''
        pass


    def store_push_example(self):
        '''
        Store an example that can be used for training later
        '''
        pass


    def get_batch(self, length):
        '''
        Get a batch of examples to train
        '''
        pass


    def dump_db(self, path):
        '''
        
        '''
        pass