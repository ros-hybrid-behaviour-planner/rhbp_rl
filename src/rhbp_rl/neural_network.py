from model_interface import AbstractDDQApproximator
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd


'''
Simple neural network
'''


class QNet(AbstractDDQApproximator):

    def __init__(self, number_inputs, number_outputs):
        super(QNet, self).__init__(number_inputs, number_outputs)
        self.lr = 0.001

    def re_init(self, number_inputs, number_outputs):
        '''
        This is a simple neural network implemented in tensorflow, tested 1.13, easily migratable to 2.0 because it uses in-built keras adaptation
        '''
        self.__model = keras.Sequential([
            layers.Dense(64, activation=tf.nn.tanh,
                         input_shape=(number_inputs,), use_bias=True, kernel_initializer=tf.initializers.glorot_uniform()),
            layers.Dense(64, activation=tf.nn.tanh,
                         kernel_initializer=tf.initializers.glorot_uniform(), use_bias=True),
            layers.Dense(64, activation=tf.nn.tanh,
                         kernel_initializer=tf.initializers.glorot_uniform(), use_bias=True),
            layers.Dropout(rate=0.2),
            layers.Dense(number_outputs)
        ])
        self.__model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.SGD(
            self.lr), metrics=['mean_squared_error'])
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
        self.__model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.SGD(
            0.0001), metrics=['mean_squared_error'])
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
        return self.__model()    

    def get_soft_update_weights(self, source, tau):
        '''
        Computes the weighted average of the weights with tau paramters, not the most efficient implementation
        '''
        target_weights = self.__model.get_weights()
        source_weights = source
        assert len(target_weights) == len(source_weights)    
        return [phi * tau for phi in source_weights] + [phi * (1. - tau) for phi in target_weights]
