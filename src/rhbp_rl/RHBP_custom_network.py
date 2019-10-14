from model_interface import AbstractDDQApproximator
import tensorflow as tf
import pandas as pd

class RHBPCustomNeuralNetwork(AbstractDDQApproximator):

    def __init__(self, num_inputs, num_outputs):
        super(RHBPCustomNeuralNetwork, self).__init__(num_inputs, num_outputs)
        self.__model = None
        tf.set_random_seed(5)
        self.step = 0

    def re_init(self, num_inputs, num_outputs):
        self.num_inputs = num_inputs
        self.num_inputs = num_outputs
        self.__model = tf.keras.Sequential()

        self.__model.add(tf.keras.layers.Dense(128 + num_outputs, tf.nn.softmax,
                                        input_shape=(self.num_inputs,), use_bias=True, kernel_initializer=tf.keras.initializers.random_uniform()))
        
        self.__model.add(tf.keras.layers.Dense(64 + num_outputs, activation=tf.nn.softmax,
                                      use_bias=True, kernel_initializer=tf.keras.initializers.random_uniform()))

        self.__model.add(tf.keras.layers.Dense(32 + num_outputs, activation=tf.nn.softmax,
                                      use_bias=True, kernel_initializer=tf.keras.initializers.random_uniform()))

        self.__model.add(tf.keras.layers.Dense(self.num_outputs))

        optimizer = tf.keras.optimizers.Adam(lr=0.001)

        self.__model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mean_squared_error'])

        print(self.__model.summary())


    def predict(self, input_state):    
        return self.__model.predict(input_state)

    def train(self, states, labels):
        
        history = self.__model.fit(states, labels, epochs=3, verbose=0)
        
        return pd.DataFrame(history.history)['mean_squared_error']

    def save_model(self, path):
        self.__model.save(path + '.h5', overwrite=True, include_optimizer=True)

    def load_model(self, path):
        self.__model = tf.keras.models.load_model(path + '.h5')
        optimizer = tf.keras.optimizers.Adam(lr=0.001)
        self.__model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mean_squared_error'])
        print(self.__model.summary())
        return True

    def get_weights_for_sync(self):
        '''
        Returns the weights for the network
        :return: weights of the network
        '''
        return self.__model.get_weights()

    def set_weights_for_sync(self, weights):
        '''
        Sets the weights for the network
        :set: weights of the network
        '''
        return self.__model.set_weights(weights)

    def sync_nets(self, to_sync, tau=0.01, hard=False):
        '''
        this method sync the networks softly via computing the updated weights from tau parameter
        '''
        self.step+=1
        if hard and (self.step % 7 == 0):
            print("Hard updated the network")
            self.hard_update(to_sync)
            self.step=0
            return
        source = to_sync.get_weights_for_sync()
        updates = self.get_soft_update_weights(source, tau)
        self.__model.set_weights(updates)

    def get_model(self):
        return self.__model

    def set_model(self, mod):
        self.__model = mod

    def get_soft_update_weights(self, source, tau):
        '''
        Computes the weighted average of the weights with tau paramters, not the most efficient implementation
        '''
        target_weights = self.__model.get_weights()
        source_weights = source
        assert len(target_weights) == len(source_weights)
        return [phi * tau for phi in source_weights] + [phi * (1. - tau) for phi in target_weights]

    def produce_target(self):
        target = RHBPCustomNeuralNetwork(self.num_inputs, self.num_outputs)
        model = tf.keras.models.clone_model(self.__model)
        target.set_model(model)
        target.hard_update(self)
        return target

    def hard_update(self, source):
        self.__model.set_weights(source.get_model().get_weights())