"""
The different configuration parameters. They get Whether here set or as ros parameters
@author: lehmann, hrabia
"""
import rospy

import utils.rhbp_logging
rhbplog = utils.rhbp_logging.LogManager(logger_name=utils.rhbp_logging.LOGGER_DEFAULT_NAME + '.rl')


class NNConfig(object):
    """
    used for setting the neural network 
    tries to get the values from the rospy param. if no connect can be made itt used defualt values
    can be extended to configure the layers of the neural network or the optimizers
    """

    def __init__(self):
        try:
            # True if AdamOptimizer should be used. Uses GradientDescentOptimizer otherwise
            self.use_adam_optimizer = rospy.get_param("~use_adam_optimizer", True)
            self.learning_rate_optimizer = rospy.get_param("~learning_rate_optimizer",
                                                           0.001)  # learning rate of the optimizer
            self.hidden_layer_amount = rospy.get_param("~hidden_layer_amount", 2)
            self.hidden_layer_cell_amount = rospy.get_param("~hidden_layer_cell_amount", 32)
            self.dropout_rate = rospy.get_param("~dropout_rate", 0.0) #dropout rate
            self.num_r_units = rospy.get_param("~num_r_units", 32) #then number of units in recurrent neural network layer
            self.activation = rospy.get_param("~activation_function", "relu") #the activation function that will be used for all layers of the network
            self.batch_norm = rospy.get_param("~batch_norm", False) #whether to use batch norm after first layer
        except Exception:  # catches if no RosService was found
            self.learning_rate_optimizer = 0.001
            self.use_adam_optimizer = True
            self.hidden_layer_amount = 1
            self.hidden_layer_cell_amount = 32
            self.dropout_rate = 0
            self.num_r_units = 32
            self.activation = "relu"
            self.batch_norm = False

        rhbplog.loginfo("use_adam_optimizer: %s", self.use_adam_optimizer)
        rhbplog.loginfo("learning_rate_optimizer: %2.3f", self.learning_rate_optimizer)
        rhbplog.loginfo("hidden_layer_amount: %d", self.hidden_layer_amount)
        rhbplog.loginfo("hidden_layer_cell_amount: %d", self.hidden_layer_cell_amount)
        rhbplog.loginfo("dropout_rate: %f", self.dropout_rate)
        rhbplog.loginfo("num_r_units : %d", self.num_r_units)
        rhbplog.loginfo("activation : %s", self.activation)


class DQNConfig(object):
    """
    sets parameter for configuring DQN
    """

    def __init__(self):
        try:
            # Set learning parameters
            self.y = rospy.get_param("~y", 0.95)  # Discount factor.
            self.tau = rospy.get_param("~tau", 0.01)  # Amount to update target network at each step.
            self.batch_size = rospy.get_param("~batch_size", 32)  # Size of training batch
            self.buffer_size = rospy.get_param("~buffer_size", 300)  # size of the experience learning buffer
            self.train_interval = rospy.get_param("~train_interval", 5)  # train the model every train_interval steps
            self.stop_training = rospy.get_param("~stop_training",
                                                 6000000)  # steps after the model does not get trained anymore
            self.pre_train = rospy.get_param("~pre_train", 36)  # no training before this many steps
            self.episodic = rospy.get_param("~episodic", False) #set to true will make the ddql gather all the episodes timesteps, apply reward and save all of them to buffer
            self.bootstrap = rospy.get_param("~bootstrap", 500) #if in episodic modes, sets the maximal number of steps to apply reward to in relative to the last step
            self.timeseries = rospy.get_param("~timeseries", False) #set to true will make the training utilise timeseries mode, requires RNN as approximator
            self.timeseries_steps = rospy.get_param("~timeseries_steps", 1) #the length of timeseries training
            self.propagate = rospy.get_param("~propagate", False) #this parameters defines if the episode reward will be propagated back towards the earliery episodes
            self.hard_update = rospy.get_param("~hard_update", False) #If true will only update target network at interval and not during each training phase
            self.sampling_rate = rospy.get_param("~sampling_rate", 3) #the number of samples drawn during trainig phase
            self.use_custom_network = rospy.get_param("~use_custom_network", False) #if true will use the custom network, if the network is not importable, will default
            self.full_buffer_training = rospy.get_param("~full_buffer_training", False) #if true will use the whole available memory buffer instead of random samples
        except Exception:
            self.y = 0.95  # Discount factor.
            self.tau = 0.01  # Amount to update target network at each step.
            self.batch_size = 32  # Size of training batch
            self.buffer_size = 10000  # size of the experience learning buffer
            self.train_interval = 5  # train the model every train_interval steps
            self.stop_training = 6000000  # steps after the model does not get trained anymore
            self.pre_train = 36  # no training before this many steps
            self.bootstrap = 500
            self.episodic = True
            self.timeseries = False #set to true will make the training utilise timeseries mode, requires RNN as approximator
            self.timeseries_steps = 8
            self.propagate = False
            self.hard_update = False
            self.success_sampling = False
            self.sampling_rate = 3
            self.use_custom_network = False
            self.full_buffer_training = False

        rhbplog.loginfo("Discount factor y: %2.3f", self.y)
        rhbplog.loginfo("Target update tau: %2.3f", self.tau)
        rhbplog.loginfo("batch_size: %d", self.batch_size)
        rhbplog.loginfo("buffer_size: %d", self.buffer_size)
        rhbplog.loginfo("train_interval: %d", self.train_interval)
        rhbplog.loginfo("stop_training: %d", self.stop_training)
        rhbplog.loginfo("pre_train: %d", self.pre_train)
        rhbplog.loginfo("episodic: %d", self.episodic)
        rhbplog.loginfo("bootstrap: %d", self.bootstrap)

        if self.batch_size > self.pre_train:
            self.pre_train = self.batch_size
            rospy.logwarn("Required: pre_train >= batch_size. Setting pre_train=%d", self.batch_size)


class SavingConfig(object):
    """
    used for saving the model 
    """

    def __init__(self):
        try:
            self.model_path = rospy.get_param("~model_path", 'models/rl-model')  # path of the saved model
            self.model_directory = rospy.get_param("~model_directory", './models')  # directory of the saved model
            self.save = rospy.get_param("~save", True)  # if the model should be saved
            self.save_buffer = rospy.get_param("~save_buffer", True)  # if the buffer should be saved
            self.steps_save = rospy.get_param("~steps_save", 10000)  # interval for saving model
            self.load = rospy.get_param("~load", False)  # if the model should be loaded
        except Exception:
            self.model_path = 'models/rl-model'  # path of the saved model
            self.model_directory = './models'  # directory of the saved model
            self.save = True  # if the model should be saved
            self.save_buffer = True  # if the buffer should be saved
            self.steps_save = 10000  # interval for saving model
            self.load = True  # if the model should be loaded


class EvaluationConfig(object):
    """
    used for saving loss error and reward plots
    """

    def __init__(self):
        try:
            self.eval_step_interval = rospy.get_param("~eval_step_interval",
                                                      500)  # interval for plotting current results
            self.eval_mean_size = rospy.get_param("~eval_mean_size",
                                                  5000)  # number of plotting mean for loss and rewards

            self.plot_rewards = rospy.get_param("~plot_rewards", False)  # if rewards should be plotted
            self.plot_loss = rospy.get_param("~plot_loss", False)  # if loss should be plotted
        except Exception:
            self.eval_step_interval = 10000  # interval for plotting current results
            self.eval_mean_size = 5000  # number of plotting mean for loss and rewards
            self.plot_rewards = True  # if rewards should be plotted
            self.plot_loss = True  # if loss should be plotted


class ExplorationConfig(object):
    """
    used for setting the parameter for the exploration strategy
    """

    def __init__(self):
        try:
            # let the model choose random actions and dont train for these number of steps
            self.pre_train = rospy.get_param("~pre_train", 000)
            self.startE = rospy.get_param("~startE", 0.50)  # set to 0 to disable exploration.
            self.endE = rospy.get_param("~endE", 0.01)
            self.annealing_steps = rospy.get_param("~annealing_steps", 100000)  # steps until it reaches endE
            # function that describes the stepDrop changing epsilon
            self.stepDrop = (self.startE - self.endE) / self.annealing_steps
        except Exception:
            self.pre_train = 000
            self.startE = 0.00
            self.endE = 0.0
            self.annealing_steps = 10000  # steps until it reaches endE
            # function that describes the stepDrop changing epsilon
            self.stepDrop = (self.startE - self.endE) / self.annealing_steps

        rhbplog.loginfo("pre_train: %d", self.pre_train)
        rhbplog.loginfo("startE: %2.3f", self.startE)
        rhbplog.loginfo("endE: %2.3f", self.endE)
        rhbplog.loginfo("annealing_steps:%d", self.annealing_steps)
        rhbplog.loginfo("stepDrop:%2.3f", self.stepDrop)


class TransitionConfig(object):
    """
    used to set parameter which set the transitioning from rhbp to the rl component
    """

    def __init__(self):
        try:
            self.use_wishes = rospy.get_param("~use_wishes", False)  # if the transformer should use wishes as input
            self.use_true_values = rospy.get_param("~use_true_values",
                                                   True)  # if the transformer should use conditions as input
            self.max_activation = rospy.get_param("~max_activation", 1)  # maximal RL activation/reinforcement
            self.min_activation = rospy.get_param("~min_activation", 0)  # minimal RL activation/reinforcement
            self.weight_rl = rospy.get_param("~weight_rl", 1.0)  # weight of the rl component
            self.use_negative_states = rospy.get_param("~use_negative_states", False)
            self.negative_reward = rospy.get_param("~negative_reward", 0)  # negative reward for negative_states
            self.use_node = rospy.get_param("~use_node", False)  # if a own node should be used for the rl_component
            self.marl_steps = rospy.get_param("~marl_steps", 0) #how many steps to consider for marl in the past
            self.reward_by_sensor = rospy.get_param("~reward_by_sensor", False)
            self.consider_marl_goals = rospy.get_param("~consider_marl_goals", False)
        except Exception:
            self.use_wishes = False
            self.use_true_values = True
            self.max_activation = 1
            self.min_activation = 0
            self.weight_rl = 1.0
            self.use_negative_states = True
            self.negative_reward = 0
            self.use_node = False
            self.marl_steps = 0
            self.reward_by_sensor = False
