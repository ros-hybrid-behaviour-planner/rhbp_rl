#! /usr/bin/env python2
"""
@author: gozman, lehman, hrabia
"""
import rospy
from dqn_model import DQNModel
from rhbp_rl.msg import ActivationState
from rhbp_rl.srv import GetActivation, GetActivationResponse
from experience import ExperienceBuffer
import numpy

import utils.rhbp_logging
rhbplog = utils.rhbp_logging.LogManager(logger_name=utils.rhbp_logging.LOGGER_DEFAULT_NAME + '.rl')

class RLComponent():
    """
    The rl component as a class. functions as a bridge between manager and rl-algo.
    It can also be used in a separated node through its service interface.
    """
    def __init__(self, name):
        """
        :param name: name of the rl_component
        """
        # name of the rl_component
        self.name = name
        # Service for communicating the activations
        self._get_activation_service = rospy.Service(name + 'GetActivation', GetActivation,
                                                     self._get_activation_state_callback)
        
        self.model = None #In this step the appropriate model will be instantiated
        # save the last state
        self.last_state = None
        self.experience = ExperienceBuffer()
        # the dimensions of the model
        self.number_outputs = -1
        self.number_inputs = -1
        self._unregistered = False
        self.buffer = ExperienceBuffer()
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
    def get_activation_state(self, input_state, negative_states=None):
        """
        Determine the activation/reinforcement for the given input states, save the state (combined with last
        state for training)
        :param input_state:
        :type input_state: InputState
        :param negative_states:
        :return: ActivationState
        """
        if negative_states is None:
            negative_states = []

        try:
            self.check_if_model_is_valid(input_state.num_inputs, input_state.num_outputs)
            # GOZMAN saving state during the getting activation for next one is not a great idea
            # TODO refactor it by decoupling this operation (might be tricky)
            if input_state.last_action:  # only save state if we have a valid prior action.
                # save current input state
                self.save_state(input_state)
                # update the last state, which would also be the starting point for the negative states
                self.last_state = input_state.input_state
                # save negative states if available
                for state in negative_states:
                    self.save_state(state, is_extra_state=True)
                # update the model
                self.train_model()#NOW THIS STUFF WILL BE ADAPTED TO THE NEW MODEL INTERFACE

            # transform the input state and get activation
            transformed_input = numpy.array(input_state.input_state).reshape(([1, len(input_state.input_state)]))
            activations = self.model.feed_forward(transformed_input)
            # return the activation via the service
            activations = activations.tolist()[0]
            activation_state = ActivationState(**{
                "name": self.name,  # this is sent for sanity check and planner status messages only
                "activations": activations,
            })
            return activation_state
        except Exception as e:
            rhbplog.logerr(e.message)
            return None
    
    #This will directly talk to the buffer
    def save_state(self, input_state, is_extra_state=False):
        pass

    #Not sure if we even need this
    def check_if_model_is_valid(self, num_inputs, num_outputs):
        pass

    #Not sure if we even need this
    def init_model(self, num_inputs, num_outputs):
        pass

    def train_model(self):
        '''
        This is a call to train model, we sample the buffer and train the model
        '''
        samples = self.buffer.get_sample(32)
        self.model.train(samples)
        pass

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
        rl_component = RLComponent(name=name)

        rospy.spin()

    except rospy.ROSInterruptException:
        rhbplog.logerr("program interrupted before completion")
