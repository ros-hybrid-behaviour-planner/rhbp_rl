from model_interface import AbstractDDQApproximator
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd


'''
Simple neural network
'''


class DefaultQNet(AbstractDDQApproximator):

    def __init__(self, number_inputs, number_outputs, use_adam=False, num_hidden_layers=2, num_neurons=16, dropout_rate=0.2, lr=0.001):
        super(DefaultQNet, self).__init__(number_inputs, number_outputs)
        assert (num_hidden_layers > 0), 'Number of layers should be bigger than 0'
        assert (num_neurons > 0), 'Number of neurons per layer should be bigger than 0'
        assert (dropout_rate < 1), 'Drop out rate should not be bigger or equalt to 1'
        self.lr = lr
        self.number_inputs = number_inputs
        self.number_outputs = number_outputs
        self.outputs = number_outputs
        self.use_adam = use_adam
        self.num_hidden_layers = num_hidden_layers
        self.num_neurons = num_neurons
        self.dropout_rate = dropout_rate
        self.__model = None

    def re_init(self, number_inputs, number_outputs):
        '''
        This function build the default neural networks with parameters that were passed upon the initialization
        '''
        self.number_inputs = number_inputs
        self.number_outputs = number_outputs
        self.__model = keras.Sequential()

        self.__model.add(layers.Dense(self.num_neurons, activation=tf.nn.tanh,
                                      input_shape=(self.number_inputs,), use_bias=True, kernel_initializer=tf.keras.initializers.random_uniform()))

        for i in range(self.num_hidden_layers-1):
            self.__model.add(layers.Dense(self.num_neurons, activation=tf.nn.tanh,
                                      use_bias=True, kernel_initializer=tf.keras.initializers.random_uniform()))
        
        
        self.__model.add(layers.Dense(self.num_neurons, activation=tf.nn.tanh,
                                      use_bias=True, kernel_initializer=tf.keras.initializers.random_uniform()))

        
        if self.dropout_rate > 0:
            self.__model.add(layers.Dropout(rate=self.dropout_rate))
        
        self.__model.add(layers.Dense(self.number_outputs))
        optimizer = None
        if self.use_adam:
            optimizer = tf.keras.optimizers.Adam(lr=self.lr)
        else:
            optimizer=tf.keras.optimizers.SGD(lr=self.lr)
        self.__model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mean_squared_error'])
        print(self.__model.summary())

    def predict(self, input_state):
        return self.__model.predict(input_state)

    def train(self, states, labels):
        history = self.__model.fit(states, labels, epochs=5, verbose=0)
        return pd.DataFrame(history.history)['mean_squared_error']

    def save_model(self, path):
        self.__model.save(path + '.h5', overwrite=True, include_optimizer=True)

    def load_model(self, path):
        self.__model = tf.keras.models.load_model(path + '.h5')
        optimizer = None
        if self.use_adam:
            optimizer = tf.keras.optimizers.Adam(lr=self.lr)
        else:
            optimizer=tf.keras.optimizers.SGD(lr=self.lr)
        self.__model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mean_squared_error'])
        print(self.__model.summary())
        return True

    def get_weights_for_sync(self):
        '''
        Returns the numpy weights for the network
        :return: weights of the network
        '''
        return self.__model.get_weights()

    def set_weights_for_sync(self, weights):
        '''
        Sets the numpy weights for the network
        :set: weights of the network
        '''
        return self.__model.set_weights(weights)

    def sync_nets(self, to_sync, tau=0.01):
        '''
        this method sync the networks softly via computing the updated weights from tau parameter
        '''
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
        target = DefaultQNet(self.number_inputs, self.number_outputs, self.use_adam, self.num_hidden_layers, self.num_neurons, self.dropout_rate, self.lr)
        model = tf.keras.models.clone_model(self.__model)
        target.set_model(model)
        return target
