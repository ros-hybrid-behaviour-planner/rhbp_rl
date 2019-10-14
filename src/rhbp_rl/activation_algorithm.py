'''
Created on 22.03.2017

@author: hrabia, lehmann, gozman
'''
from __future__ import division  # force floating point division when using plain /

import rospy

import subprocess
import numpy as np
from rhbp_core.msg import Activation
from exploration_strategies import EpsilonGreedyPreTrain
from input_state_transformer import InputStateTransformer
from rl_config import TransitionConfig

from rhbp_rl.msg import InputState
from rhbp_rl.srv import GetActivation
from rl_component import RLComponent
from behaviour_components.activation_algorithm import BaseActivationAlgorithm, ActivationAlgorithmFactory

import utils.rhbp_logging
rhbplog = utils.rhbp_logging.LogManager(
    logger_name=utils.rhbp_logging.LOGGER_DEFAULT_NAME + '.rl')


class ReinforcementLearningActivationAlgorithm(BaseActivationAlgorithm):
    """
    This activation algorithm changes the activation calculation formulas in respect to the base algorithm with
    including the reinforcement learning 
    """

    def __init__(self, manager, extensive_logging=False, create_log_files=False):
        super(ReinforcementLearningActivationAlgorithm, self).__init__(
            manager, extensive_logging=extensive_logging)

        # saves list with the activations
        self.activation_rl = []
        # timeout until trying to reach the service gets stopped
        self.SERVICE_TIMEOUT = 5
        self.config = TransitionConfig()
        # values how much the rl activation influences the other components
        self.weight_rl = self.config.weight_rl
        # step counter is used for exploration
        self._step_counter = 0
        # transforms rhbp values to abstract rl values
        self.input_transformer = InputStateTransformer(manager)
        # the rl component
        self.rl_component = None
        # address of the rl node
        self.rl_address = self._manager.prefix.replace("/Manager", "_rl_node")
        # start only rl_component if needed
        if self.weight_rl > 0.0:
            # select if a own node should be started for the rl_component
            if self.config.use_node:
                rhbplog.loginfo("Starting external RL node.")
                self.start_rl_node()
                rhbplog.loginfo("External RL node started.")
            else:
                self.start_rl_class()
                rhbplog.loginfo("Using RL component directly.")
        # implements different exploration strategies
        self.exploration_strategies = EpsilonGreedyPreTrain()

        self.explored_in_last_step = False
    
    def compute_behaviour_activation_step(self, ref_behaviour):
        """
        computes out of the different activation influences the activation step. 
        :param ref_behaviour: for which behaviour the activation should be computed
        :return: 
        """
        # calling base method for getting other activations
        current_activation_step = super(ReinforcementLearningActivationAlgorithm, self).\
            compute_behaviour_activation_step(ref_behaviour=ref_behaviour)
        # get activation from reinforcement learning
        rl_activation = self.get_rl_activation_for_ref(ref_behaviour)

        rhbplog.loginfo("\t%s: activation from rl: %2.3f",
                        ref_behaviour, rl_activation)

        ref_behaviour.current_activation_step = current_activation_step + rl_activation

        ref_behaviour.activation_components.append(
            Activation('RL', rl_activation))

        return current_activation_step

    def get_rl_activation_for_ref(self, ref_behaviour):
        index = self.input_transformer.behaviour_to_index(ref_behaviour)
        # if the activation is not yet received the values are zero
        if len(self.activation_rl) == 0:
            value = 0
        else:
            value = self.activation_rl[index] #this is where the value are first actually used
            # normalisation of behaviour reinforcement activations
            max_val = np.max(self.activation_rl)
            min_val = np.min(self.activation_rl)
            b = self.config.max_activation
            a = self.config.min_activation
            value = (b - a) * (value - min_val) / (max_val - min_val) + a
            value *= self.config.weight_rl
        return value

    def step_preparation(self):
        """
        overrides step_preparation. includes fetching the most recent activation for the input state and
        choosing a random action according to the exploration strategy 
        """
        # include BaseActivation functions
        super(ReinforcementLearningActivationAlgorithm, self).step_preparation()

        # TODO Idea we could also consider to move the rest of this method to a future/thread to increase concurrency
        num_actions = len(self._manager.behaviours)
        # if the rl activation is not used dont calculate the values and set all to zero
        if self.weight_rl <= 0:  
            self.activation_rl = [0] * num_actions  # set all activations to 0
            return

        # get the activations from the rl_component
        activation_rl = self.get_activation_from_rl_node()
        # return if no activations received
        if activation_rl == None or len(activation_rl) == 0:
            rhbplog.logerr('RL was none')            
            return

        # TODO Idea: Why not test first for exploration? Test to move it before get_activation_from_rl_node. 
        # Getting activation is also saving the current state
        # hence it has to be called but maybe we can avoid the feed forward through a flag in the service?

        #Commentary: feed-forward itself does not really take that much time when compared to training that also 
        # is triggered during getting activation. 

        # check if exploration chooses randomly best action
        changed, best_action = self.exploration_strategies.get_strategy(
            self._step_counter, num_actions)
        if changed:
            activation_rl = [0] * num_actions  # set all activations to 0
            # only support the one to be explored
            activation_rl[best_action] = self.config.max_activation
            self.explored_in_last_step = True
        else:
            self.explored_in_last_step = False
        # set the step counter higher
        self.activation_rl = activation_rl
        self._step_counter += 1

    def get_activation_from_rl_node(self):
        """
        this function gets first the InputState from the rhbp components and sends this to the rl_node via service.

        :return: the received activations
        """
        # get the input state. also check if the input state is correct
        is_correct, input_state_msg = self.get_and_validate_input()
        # if input state is incorrect return
        if not is_correct:
            return
        # create an input state message for sending it to the rl_node
        

        # collect negative states (not executable behaviors)
        num_actions = len(self._manager.behaviours)
        negative_states = []
        if self.config.use_negative_states:
            for action_index in range(num_actions):
                # if random chosen number is not executable, give minus reward and sent it to rl component
                if not self._manager.behaviours[action_index].executable:
                    negative_state = InputState()
                    negative_state.input_state = input_state_msg.input_state
                    negative_state.num_outputs = input_state_msg.num_outputs
                    negative_state.num_inputs = input_state_msg.num_inputs
                    negative_state.reward = self.config.negative_reward
                    negative_state.last_action = action_index
                    negative_states.append(negative_state)

        # start service to get the activations from the model
        return self.fetch_activation(input_state_msg, negative_states)
        

    def get_and_validate_input(self):
        """
        receive via the input state transformer the rl values needed for the algorithm.
        checks if values are incomplete
        :return: returns the input state values or a False value if values are incomplete
        """
        # receive outputs from the known behaviours
        num_outputs = len(self._manager.behaviours)
        if num_outputs == 0:
            return False, None
        # receive the input values from the sensors and conditions
        input_state = self.input_transformer.transform_input_values()
        num_inputs = input_state.shape[0]
        if num_inputs == 0:
            return False, None
        # receive the reward from the goals
        reward = self.input_transformer.calculate_reward()
        # receive the executed action from the executed behaviour
        last_action = self._manager.executed_behaviours
        if len(last_action) == 0:
            # None will result in only feeding forward from the current situation.
            last_action_index = None
        else:
            last_action_index = self.input_transformer.behaviour_to_index(
                last_action[0]) 
            if last_action_index is None:
                return False, None
        
        input_state_msg = InputState()
        input_state_msg.input_state = input_state
        input_state_msg.num_outputs = num_outputs
        input_state_msg.num_inputs = num_inputs
        input_state_msg.reward = reward
        input_state_msg.last_action = last_action_index


        return True, input_state_msg

    def fetch_activation(self, input_state_msg, negative_states):
        """
        This method fetches the activation from the RL component either directly or through
        rl node via GetActivation service call
        :param input_state:
        :param negative_states: 
        :return: 
        """
        if self.config.use_node:  # either use the service or a direct method call.
            try:
                rhbplog.logdebug("Waiting for service %s",
                                 self.rl_address + 'GetActivation')
                rospy.wait_for_service(
                    self.rl_address + 'GetActivation', timeout=self.SERVICE_TIMEOUT)
            except rospy.ROSException:
                rospy.logerr("GetActivation service not found")
                return
            try:
                if input_state_msg.last_action == None:
                   input_state_msg.last_action = 0 #other service with complain about type
                get_activation_request = rospy.ServiceProxy(
                    self.rl_address + 'GetActivation', GetActivation)
                activation_state = get_activation_request(
                    input_state_msg, negative_states).activation_state

            except rospy.ServiceException as e:
                rhbplog.logerr(
                    "ROS service exception in 'fetchActivation' of behaviour '%s':", self.rl_address)
        else:
            activation_state = self.rl_component.get_activation_state(
                input_state_msg, negative_states)

        if activation_state:
            return list(activation_state.activations)
        else:
            return []

    def start_rl_class(self):
        """
        starts the rl_component as a class
        :return: 
        """
        rhbplog.loginfo("starting rl_component as a class")
        self.rl_component = RLComponent(name=self.rl_address, prefix = self._manager.prefix)

    def start_rl_node(self):
        """
        start the rl_component as a node
        """
        rospy.logwarn("starting rl_component as a node")
        package = 'rhbp_rl'
        executable = 'src/rhbp_rl/rl_component.py'
        command = "rosrun {0} {1} _name:={2} _prefix:={3}".format(
            package, executable, self.rl_address, self._manager.prefix)
        p = subprocess.Popen(command, shell=True)

        state = p.poll()
        if state is None:
            rhbplog.loginfo("RL process is running fine")
        elif state < 0:
            rhbplog.loginfo("RL Process terminated with error")
        elif state > 0:
            rhbplog.loginfo("RL Process terminated without error")
        rospy.sleep(2)



    

def register_in_factory(activation_algo_factory):
    """
    After this module has been imported you have to register the algos from this module in the other factory by
    calling this method.
    :param activation_algo_factory: the factory to register the activation algorithm
    :type activation_algo_factory: ActivationAlgorithmFactory
    """

    activation_algo_factory.register_algorithm(
        "reinforcement", ReinforcementLearningActivationAlgorithm)
