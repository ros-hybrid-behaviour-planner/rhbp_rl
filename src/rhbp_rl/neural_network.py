from model_interface import AbstractDDQApproximator
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd

'''
This is just a temporary nn to use for debgging, basically taken from tf-2.0 tutorials, nothing fancy
'''

class BaseNet1(AbstractDDQApproximator):

    def __init__(self, number_inputs, number_outputs):
        super(BaseNet1, self).__init__(number_inputs, number_outputs)
        print(number_inputs)
        self.__model = keras.Sequential([
            layers.Dense(64, activation=tf.nn.relu,
                         input_shape=[number_inputs]),
            layers.Dense(64, activation=tf.nn.relu),
            layers.Dense(number_outputs)
        ])
        self.__model.compile(loss='mean_squared_error', optimizer=tf.optimizers.RMSprop(
            0.0001), metrics=['mean_squared_error'])
        print(self.__model.summary())

    
    
    
    def re_init(self, number_inputs, number_outputs):
        self.__model = keras.Sequential([
            layers.Dense(64, activation=tf.nn.relu,
                         input_shape=[1, number_inputs]),
            layers.Dense(64, activation=tf.nn.relu),
            layers.Dense(number_outputs)
        ])
        self.__model.compile(loss='mean_squared_error', optimizer=tf.optimizers.RMSprop(
            0.001), metrics=['mean_squared_error'])

    
    
    def predict(self, input_state):
        return self.__model.predict(input_state)

    
    
    def train(self, states, labels):
        history = self.__model.fit(states, labels, epochs=10)
        return pd.DataFrame(history.history)['mean_squared_error']

    def save_model(self, path):
        self.__model.save(path, include_optimizer=True)


    def load_model(self, path):
        self.__model.load_weights(path)
        return True

    def get_weights_for_sync(self):
        return self.__model.get_weights()

    def set_weights_for_sync(self, weights):
        return self.__model.set_weights(weights)

    