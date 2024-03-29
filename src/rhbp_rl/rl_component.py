#! /usr/bin/env python2
"""
@author: lehmann, hrabia, gozman
"""
import rospy
from ddql import DDQLAlgo
from rhbp_rl.msg import ActivationState
from rhbp_rl.srv import GetActivation, GetActivationResponse, EpEnded
import numpy
import tensorflow
from rl_sensors_client import get_b_index
import traceback

import utils.rhbp_logging
rhbplog = utils.rhbp_logging.LogManager(logger_name=utils.rhbp_logging.LOGGER_DEFAULT_NAME + '.rl')


class RLComponent(object):
    """
    The rl component as a class. functions as a bridge between manager and rl-algo.
    It can also be used in a separated node through its service interface.
    """

    def __init__(self, name, prefix=""):
        """
        :param name: name of the rl_component
        """
        # name of the rl_component
        self.name = name
        # True if the model was set up
        self.is_model_init = False
        # Service for communicating the activations
        self._get_activation_service = rospy.Service(prefix + '/GetActivation', GetActivation,
                                                     self._get_activation_state_callback)
        
        self._get_episode_end_service = rospy.Service(prefix + '/EpisodeEnd', EpEnded,
                                                    self._get_episode_end_callback)

        # choose appropriate model
        self.model = DDQLAlgo(self.name)

        # save the last state
        self.last_state = None
        # the dimensions of the model
        self.number_outputs = -1
        self.number_inputs = -1
        self._unregistered = False
        self.episode_ended = False
        self.reward = 0
        self.prefix = prefix
        rospy.on_shutdown(self.unregister)  # cleanup hook also for saving the model.

    def _get_activation_state_callback(self, request_msg):
        """
        answers the RL activation service and responds with the activations/reinforcements
        :param request_msg: GetActivation 
        :return: Service Response
        """
        input_state = request_msg.input_state
        negative_states = request_msg.negative_states
        try:
            activation_state = self.get_activation_state(input_state, negative_states)
            return GetActivationResponse(activation_state)
        except Exception as e:
            rhbplog.logerr(e.message)
            return None
    def _get_episode_end_callback(self, episode_end_msg):
        rhbplog.loginfo("Received the call to end episode wtih reward " + str(episode_end_msg.reward))
        reward = episode_end_msg.reward
        last_b = episode_end_msg.behaviour
        action = get_b_index(last_b,self.prefix + '/GetIndex').index
        last = numpy.array(self.last_state).reshape(([1, len(self.last_state)]))
        new = None
        reward_tuple = (last, new, action, reward)
        self.model.add_sample(tuple=reward_tuple, end=True, reward_punish=reward, consider_reward=True)
        return 1

    def get_activation_state(self, input_state, negative_states=None):
        """
        Determine the activation/reinforcement for the given input states, save the state (combined with last
        state for training)
        :param input_state:
        :type input_state: InputState
        :param negative_states:
        :return: 
        ActivationState
        """
        if negative_states is None:
            negative_states = []

        try:
            self.check_if_model_is_valid(input_state.num_inputs, input_state.num_outputs)
            if input_state.last_action is not None:  # only save state if we have a valid prior action.
                # save current input state
                self.save_state(input_state, self.episode_ended, self.reward)
                # update the last state, which would also be the starting point for the negative states
                self.last_state = input_state.input_state
                # save negative states if available
                #for state in negative_states:
                #    self.save_state(state, is_extra_state=True)
                # update the model
                
                self.model.train_model()

            # transform the input state and get activation
            transformed_input = numpy.array(input_state.input_state).reshape(([1, len(input_state.input_state)]))
            activations = self.model.predict(transformed_input)
            # return the activation via the service
            activations = activations.tolist()[0]
            activation_state = ActivationState(**{
                "name": self.name,  # this is sent for sanity check and planner status messages only
                "activations": activations,
            })
            self.episode_ended = False
            self.reward = 0
            return activation_state
        except Exception as e:
            self.episode_ended = False
            self.reward = 0
            rhbplog.logerr(e.message)
            rhbplog.logerr(traceback.format_exc())
            exit()
            return None

    def save_state(self, input_state, end, reward_punish, is_extra_state=False):
        """
        save the old_state,new_state,action,reward tuple for batch updating of the model
        :param input_state: current state input (positive or negative)
        :type input_state: InputState
        :param is_extra_state: set to True if this is a special extra state (e.g. negative states) that is recorded but
                               not necessarily has been explored/executed
        """
        if self.last_state is None:
            return
        last = numpy.array(self.last_state).reshape(([1, len(self.last_state)]))
        new = numpy.array(input_state.input_state).reshape(([1, len(input_state.input_state)]))
        reward_tuple = (last, new, input_state.last_action, input_state.reward)
        self.model.add_sample(tuple=reward_tuple, end=end, reward_punish=reward_punish, consider_reward=not is_extra_state)
        


    def check_if_model_is_valid(self, num_inputs, num_outputs):
        """
        checks if the in-/outputs are the same as the current model has. If not 
        a new model is started
        :param num_inputs: 
        :param num_outputs: 
        :return: 
        """
        if not self.is_model_init:
            self.init_model(num_inputs, num_outputs)
        else:
            if (not self.number_outputs == num_outputs) or (not self.number_inputs == num_inputs):
                self.init_model(num_inputs, num_outputs)

    def init_model(self, num_inputs, num_outputs):
        """
        inits the model with the specified parameters
        :param num_inputs: 
        :param num_outputs: 
        :return: 
        """
        self.number_inputs = num_inputs

        self.number_outputs = num_outputs

        self.last_state = None

        self.model.start_nn(num_inputs, num_outputs)
        
        self.is_model_init = True

    def unregister(self):
        if not self._unregistered:
            self._unregistered = True
            if self.model:
                self.model.save_model()

    def __del__(self):
        self.unregister()


# Component can also be started as a independent node.
if __name__ == '__main__':
    try:
        rospy.init_node('rl_node', anonymous=True)
        name = rospy.get_param("~name", "rl_component_node")
        prefix = rospy.get_param("~prefix", "")
        rl_component = RLComponent(name=name, prefix=prefix)
        rospy.logwarn("Starting node")
        rospy.spin()

    except rospy.ROSInterruptException:
        rhbplog.logerr("program interrupted before completion")
