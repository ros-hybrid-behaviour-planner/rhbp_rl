"""
class for implementing the dqn model. including savign the model,
metrics for measuring the success, the experience buffer and neural network
@author: lehmann, hrabia, gozman
"""

from __future__ import division
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
        self.training = False
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
        # Number of steps used before training updates begin.
        self.pre_train_steps = self.model_config.pre_train
        self.q_net = None
        self.target_net = None
        # buffer class for experience learning
        self.exp_buffer = ExperienceBuffer(self.model_config.buffer_size, self.model_config.timeseries)
        self.model_training_counter = 0
        self.saver = None
        self.reward_saver = []
        self.loss_over_time = []
        self.rewards_over_time = []
        self.num_inputs = 0
        self.num_outputs = 0
        self.episode_run = []
        # model variables

    def start_nn(self, num_inputs, num_outputs):
        """
          calls to start the neural network. checks first if one already exists.
          :param num_inputs: 
          :param num_outputs: 
          :return: 
          """
        self.exp_buffer.reset(True)
        if not os.path.exists(self.model_folder):
            os.makedirs(self.model_path)
        self.model_path = self.model_path + str(num_inputs) + '-' + str(self.nn_config.hidden_layer_amount) + '-' + str(
            self.nn_config.hidden_layer_cell_amount) + '-' + str(self.nn_config.dropout_rate)
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs    
        try:
            from RHBP_custom_network import RHBPCustomNeuralNetwork
        except ImportError:
            rhbplog.logwarn("Could not import custom neural network")
            self.model_config.use_custom_network = False
        if not self.model_config.use_custom_network:     
            self.q_net = DefaultQNet(num_inputs, num_outputs, self.nn_config.use_adam_optimizer,
                                    self.nn_config.hidden_layer_amount,
                                    self.nn_config.hidden_layer_cell_amount,
                                    self.nn_config.dropout_rate,
                                    self.nn_config.learning_rate_optimizer,
                                    self.nn_config.activation,
                                    self.model_config.timeseries,
                                    self.nn_config.batch_norm)
            rhbplog.logwarn("Created network")
        else:
            self.q_net = RHBPCustomNeuralNetwork(num_inputs, num_outputs)        
        if self.save_config.load:
            try:
                self.q_net.load_model(self.model_path)
                rhbplog.logwarn("Loaded checkpoint")
            except Exception as e:
                rhbplog.logerr(
                    "Failed loading model, initialising a new one. Error: %s", e)
                self.q_net.re_init(num_inputs, num_outputs)
        else:
            rhbplog.logwarn("Load parameter is false, initiating new neural network")
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



    def add_to_buffer(self, reward_punish):
        """
        Preprocesses examples and adds them to buffer when no discount is needed for episodic mode
        """
        if self.model_config.propagate:                 
            for i in range(len(self.episode_run)):
                self.episode_run[i][0][2] = reward_punish
                self.exp_buffer.add(self.episode_run[i])
        else:
            for i in range(len(self.episode_run)):
                self.exp_buffer.add(self.episode_run[i])

    def add_sample(self, tuple, end=False, reward_punish = 0, consider_reward=True):
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

        # save the input tuple in buffer if in non-episodic mode
        transformed_tuple = np.reshape(
            np.array([tuple[0], tuple[2], tuple[3], tuple[1]]), [1, 4])

        #if in episodic mode, save to episode buffer
        if self.model_config.episodic or self.model_config.timeseries:
            self.episode_run.append(transformed_tuple)
            #save trajectory to memory the episode has ended           
            if end:
                if not self.model_config.timeseries: #if no timeseries modes is turned on, preprocess and add trajectory to buffer, otherwise add the whole episode as one sample
                    self.add_to_buffer(reward_punish)
                else:
                    if self.model_config.propagate: #propagate the reward to tuples in episode 
                        for i in range(len(self.episode_run)):
                            self.episode_run[i][0][2] = reward_punish
                        self.exp_buffer.add(self.episode_run)     
                    else:
                        self.exp_buffer.add(self.episode_run)                                       
                rhbplog.loginfo("Episode ended, gathering data... There have been %d steps, the final reward was %d"%(len(self.episode_run), reward_punish))
                self.episode_run = []
            else:
                return
        else:
            self.exp_buffer.add(transformed_tuple)



    def prepare_examples(self, full_buffer):
        '''
        Prepares examples for training by computing inputs and outputs when no timeseries mode is turned on
        :param full_buffer: make the function prepare one batch with all the available memory
        :return: gathered states with corresponding target labels
        '''
        if not full_buffer:
            train_batch = self.exp_buffer.sample(self.model_config.batch_size)
        else:
            rhbplog.logdebug("Full buffer trarnig mode on")
            train_batch = self.exp_buffer.sample(self.exp_buffer.get_length())
        states = train_batch[:,0]
        target_q = np.array([1])
        for tup in train_batch:
            if tup[3] is None:
                target_q = np.append(target_q, tup[2])
            else:
                Q1 = self.q_net.predict(tup[3])   
                Q2 = self.target_net.predict(tup[3])  
                index = np.argmax(Q2)  
                target_q = np.append(target_q, tup[2] + (self.model_config.y * Q1[:,index]))        
        target_q = np.delete(target_q, 0)
        one_hots1 = np.array(train_batch[:,1], dtype=np.int32).reshape(-1)
        one_hots = np.eye(self.num_outputs)[one_hots1]
        target_q_labels = np.multiply(one_hots, np.array(
            [target_q, ]*self.num_outputs).transpose())
        return states, target_q_labels


    def prepare_examples_for_timeseries(self):
        '''
        Prepares examples for training by computing inputs and outputs when  timeseries mode is turned on. short batches with ordered tuples
        :return: gathered states with corresponding target labels
        '''
        train_batch = self.exp_buffer.sample(1, self.model_config.timeseries_steps)
        if len(train_batch) < 1:
            return None, None
        states = train_batch[:,0]
        target_q = np.array([1])
        for tup in train_batch:
            if tup[3] is None:
                target_q = np.append(target_q, tup[2])
            else:
                Q1 = self.q_net.predict(tup[3])   
                Q2 = self.target_net.predict(tup[3])  
                index = np.argmax(Q1)  
                target_q = np.append(target_q, tup[2] + (self.model_config.y * Q2[:,index]))        
        target_q = np.delete(target_q, 0)
        one_hots1 = np.array(train_batch[:,1], dtype=np.int32).reshape(-1)
        one_hots = np.eye(self.num_outputs)[one_hots1]
        target_q_labels = np.multiply(one_hots, np.array(
            [target_q, ]*self.num_outputs).transpose())
        for i in range (len(states)):
            states[i] = np.expand_dims(states[i], axis=1)
        return states, target_q_labels

        

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
        if self.exp_buffer.get_length() < self.pre_train_steps or self.model_training_counter % self.model_config.train_interval != 1 \
                or self.model_training_counter > self.model_config.stop_training:
            return
        rhbplog.logdebug("-----Training------")
        if not self.training:
            rhbplog.logwarn("-------------------Training has started-------------")
            self.training = True     
        if not self.model_config.timeseries:
            if not self.model_config.full_buffer_training:
                for _ in range(self.model_config.sampling_rate):
                    states, target_q_labels = self.prepare_examples(False)
                    loss = self.q_net.train(np.vstack(states), target_q_labels)              
                    # save the loss function value (squared error from q and target value)
                    self.loss_over_time.append(loss)
                    # update the target network
                    rhbplog.loginfo("Syncing the target and q-network")
                    self.target_net.sync_nets(self.q_net, self.model_config.tau, self.model_config.hard_update)

            else:
                states, target_q_labels = self.prepare_examples(self.model_config.full_buffer_training)
                loss = self.q_net.train(np.vstack(states), target_q_labels)             
                self.target_net.sync_nets(self.q_net, self.model_config.tau, self.model_config.hard_update)
        else: 
            for _ in range(int(self.model_config.batch_size/self.model_config.timeseries_steps)*self.model_config.sampling_rate): #due to the actual batches being shorter the baseline sampling is expanded
                states, target_q_labels = self.prepare_examples_for_timeseries()
                if states is None or target_q_labels is None:
                    continue
                loss = self.q_net.train(np.vstack(states), target_q_labels)
                # save the loss function value (squared error from q and target value)
                self.loss_over_time.append(loss)
                # update the target network
                rhbplog.loginfo("Syncing the target and q-network")
            self.target_net.sync_nets(self.q_net, self.model_config.tau, self.model_config.hard_update)


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
