"""
class for implementing the dqn model. including savign the model,
metrics for measuring the success, the experience buffer and neural network
@author: lehmann, hrabia
# inspired by awjuliani
"""

import utils.rhbp_logging
from experience import ExperienceBuffer
from rl_config import NNConfig, EvaluationConfig, SavingConfig, DQNConfig, ExplorationConfig
from neural_network import DefaultQNet
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle
import random
import numpy as np
import numpy
import matplotlib
import rospy
from collections import deque

import pandas as pd

matplotlib.use('agg')


rhbplog = utils.rhbp_logging.LogManager(
    logger_name=utils.rhbp_logging.LOGGER_DEFAULT_NAME + '.rl')


class DDQLAlgo():
    def __init__(self, name):

        # Set learning parameters
        self.model_config = DQNConfig()
        self.save_config = SavingConfig()
        self.save_conf = SavingConfig()
        self.nn_config = NNConfig()
        self.model_folder = self.save_conf.model_directory
        self.model_path = self.model_folder + '/' + name
        self.evaluation = Evaluation(self.model_folder)
        self.name = name
        self.eval_config = EvaluationConfig()
        self.exploration_config = ExplorationConfig()
        self.train_interval = self.model_config.train_interval
        # Number of steps used before training updates begin.
        self.pre_train_steps = self.model_config.pre_train
        self.q_net = None
        self.target_net = None
        # buffer class for experience learning
        self.exp_buffer = ExperienceBuffer(self.model_config.buffer_size)
        self.model_training_counter = 0
        self.saver = None
        self.reward_saver = []
        self.loss_over_time = []
        self.rewards_over_time = []
        self.num_inputs = 0
        self.num_outputs = 0
        # model variables

    def start_nn(self, num_inputs, num_outputs):
        """
          calls to start the neural network. checks first if one already exists.
          :param num_inputs: 
          :param num_outputs: 
          :return: 
          """
        self.exp_buffer.reset()
        if not os.path.exists(self.model_folder):
            os.makedirs(self.model_path)
        self.model_path = self.model_path + '-' + str(self.nn_config.hidden_layer_amount) + '-' + str(
            self.nn_config.hidden_layer_cell_amount) + '-' + str(self.nn_config.dropout_rate)
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.q_net = DefaultQNet(num_inputs, num_outputs, self.nn_config.use_adam_optimizer,
                                 self.nn_config.hidden_layer_amount,
                                 self.nn_config.hidden_layer_cell_amount,
                                 self.nn_config.dropout_rate)
        try:
            self.q_net.load_model(self.model_path)
            rhbplog.loginfo("Loaded checkpoint")
        except Exception as e:
            rhbplog.logerr(
                "Failed loading model, initialising a new one. Error: %s", e)
            self.q_net.re_init(num_inputs, num_outputs)
        self.target_net = self.q_net.produce_target()

    def save_model(self):
        """
        saves the model 
        :return: 
        """
        if self.q_net == None:
            return
        if not self.save_conf.save:
            return

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        self.q_net.save_model(self.model_path)

        if self.save_conf.save_buffer:
            self.save_buffer()

        rhbplog.loginfo("Saved model '%s'", self.model_path)

    def predict(self, input_state):
        """
        feed forwarding the input_state into the network and getting the calculated activations
        :param input_state: the input state as a vector
        :return: vector of activations
        """
        return self.q_net.predict(input_state)

    def load_buffer(self):
        """
        loading the experience buffer
        :return: 
        """
        size = self.model_config.buffer_size
        filename = self.model_folder + "/buffer_" + str(size) + ".txt"
        try:
            with open(filename, "rb") as fp:
                buffer = pickle.load(fp)

            self.exp_buffer.buffer = deque(buffer)
            rhbplog.loginfo("experience buffer successfully loaded")
        except Exception:
            rhbplog.loginfo(
                "File not found. Cannot load the experience buffer")

    def save_buffer(self):
        """
        saving the experience buffer
        :return: 
        """
        size = self.exp_buffer.buffer_size
        buffer = self.exp_buffer.buffer
        filename = self.model_folder + "/buffer_" + str(size) + ".txt"
        with open(filename, "wb") as fp:
            pickle.dump(buffer, fp)

    def add_sample(self, tuple, consider_reward=True):
        """
        inserting a tuple containing the chosen action in a specific situation with the resulting reward.
        see also super class description.
        :param tuple:
        :param consider_reward:
        :return:
        """

        # save rewards
        if consider_reward:
            self.rewards_over_time.append(tuple[3])

        # save the input tuple in buffer
        transformed_tuple = np.reshape(
            np.array([tuple[0], tuple[2], tuple[3], tuple[1]]), [1, 4])
        self.exp_buffer.add(transformed_tuple)


    def train_model(self):
        # check if evaluation plots should be saved after configured number of trainings
        if self.model_training_counter % self.eval_config.eval_step_interval == 0:
            if self.eval_config.plot_loss:
                self.evaluation.plot_losses(self.loss_over_time)
            if self.eval_config.plot_rewards:
                self.evaluation.plot_rewards(self.rewards_over_time)

        # check if model should be saved after configured number of trainings
        if self.model_training_counter % self.save_config.steps_save == 0:
            self.q_net.save_model(self.model_path)

        self.model_training_counter += 1

        # check if batch training should be executed
        if self.model_training_counter < self.pre_train_steps or self.model_training_counter % self.train_interval != 1 \
                or self.model_training_counter > self.model_config.stop_training:
            return

        # We use Double-DQN training algorithm
        # get sample of buffer for training
        train_batch = self.exp_buffer.sample(self.model_config.batch_size)
        Q1 = self.q_net.predict(np.vstack(train_batch[:, 3]))
        Q2 = self.target_net.predict(np.vstack(train_batch[:, 3]))
        indexes = np.argmax(Q1, 1)
        # multiplier to add if the episode ended
        # makes reward 0 if episode ended. simulation specific
        # target-q-values of batch for choosing prediction of q-network
        # target_q-values for the q-net predicted action
        double_q = Q2[range(self.model_config.batch_size), indexes]
        # target q value calculation according to q-learning
        target_q = train_batch[:, 2] + (self.model_config.y * double_q)
        actions = train_batch[:, 1]
        one_hots1 = np.array(actions, dtype=np.int32).reshape(-1)
        one_hots = np.eye(self.num_outputs)[one_hots1]
        target_q_labels = np.multiply(one_hots, np.array(
            [target_q, ]*self.num_outputs).transpose())
        loss = self.q_net.train(np.vstack(train_batch[:, 0]), target_q_labels)
        # save the loss function value (squared error from q and target value)
        self.loss_over_time.append(loss)
        # update the target network
        rhbplog.loginfo("Syncing the target and q-network")
        self.target_net.sync_nets(self.q_net, self.model_config.tau)


class Evaluation(object):
    """
    for saving metrics which can measure the success of the learning process
    """

    def __init__(self, model_folder):
        self.model_folder = model_folder
        self.eval_config = EvaluationConfig()

    def plot_rewards(self, rewards_over_time):
        """
        plots and saves the rewards over time and the mean rewards
        """
        if not os.path.exists(self.model_folder):
            os.makedirs(self.model_folder)
        if len(rewards_over_time) == 0:
            return
        # reward
        df = pd.DataFrame(numpy.array(rewards_over_time), columns=["rewards"])
        df.plot(style="o")
        plt.xlabel("steps")
        plt.savefig(self.model_folder + "/rewards_plot.png")
        plt.close()
        # mean rewards
        means = []
        batch = self.eval_config.eval_mean_size
        for i in range(0, len(rewards_over_time) / batch):
            means.append(numpy.mean(numpy.array(rewards_over_time)
                                    [batch * i:batch * (i + 1)]))
        if len(means) == 0:
            return
        df = pd.DataFrame(means, columns=["mean_rewards"])
        df.plot()
        plt.xlabel("steps")
        plt.savefig(self.model_folder + "/means_rewards_plot.png")
        plt.close()

    def plot_losses(self, loss_over_time):
        """
        plots and saves the loss error function. 
        :return: 
        """
        if not os.path.exists(self.model_folder):
            os.makedirs(self.model_folder)
        if len(loss_over_time) == 0:
            return

        # loss error
        df = pd.DataFrame(numpy.array(loss_over_time), columns=["loss error"])
        df.plot(legend=False)
        plt.xlabel("training steps")
        plt.ylabel("loss error")
        plt.savefig(self.model_folder + "/loss_error_plot.png")
        plt.close()

        # mean loss
        means = []
        batch = self.eval_config.eval_mean_size
        for i in range(0, len(loss_over_time) / batch):
            means.append(numpy.mean(numpy.array(loss_over_time)
                                    [batch * i:batch * (i + 1)]))
        if len(means) == 0:
            return
        df = pd.DataFrame(means, columns=["mean loss error"])
        df.plot()
        plt.xlabel("steps")
        plt.savefig(self.model_folder + "/means_loss_error_plot.png")
        plt.close()
