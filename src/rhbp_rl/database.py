class Database():
    '''
    Database class to provide storage for examples for training.
    Experience buffer that will be also a service potentially
    '''
    def __init__(self):
        self.examples = []
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
        :param length: the number of examples in batch
        '''
        pass


    def dump_db(self, path):
        '''
        Dumps the gathered expierience in a suitable file form that allow it to be read from
        :param path: where to save the database
        '''
        pass


    def load_db(self, path):
        '''
        Load database from the file 
        :param path: the path to file with saved db material
        '''
        pass

    def clean_db(self):
        '''
        Removes all examples from database
        '''
        self.examples = []

    