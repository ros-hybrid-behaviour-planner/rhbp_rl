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

    def re_init(self, number_inputs, number_outputs):
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
            0.0001), metrics=['mean_squared_error'])
        print(self.__model.summary())

    def predict(self, input_state):
        return self.__model.predict(input_state)

    def train(self, states, labels):
        history = self.__model.fit(states, labels, epochs=5)
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
        return self.__model.trainable_weights + sum([l.non_trainable_weights for l in self.__model.layers], [])

    def set_weights_for_sync(self, weights):
        return self.__model.set_weights(weights)

    def sync_nets(self, to_sync, tau=0.01):
        print('nigga')
        source = to_sync.get_weights_for_sync()
        print('nigga')
        updates = self.get_soft_target_model_updates(source, tau)
     
        self.__model.set_weights(to_sync.get_weights_for_sync())


    def get_model(self):
        return self.__model()    

    def get_soft_target_model_updates(self, source, tau):
        print('nigga')
        target_weights = self.__model.trainable_weights + sum([l.non_trainable_weights for l in self.__model.layers], [])
        source_weights = source
        assert len(target_weights) == len(source_weights)
        # Create updates.
        updates = []
        for tw, sw in zip(target_weights, source_weights):
            updates.append((tw, tau * sw + (1. - tau) * tw))
        print(updates)
        return updates