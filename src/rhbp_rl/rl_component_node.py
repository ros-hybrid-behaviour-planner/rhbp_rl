#! /usr/bin/env python2
"""
the rl component as a node. functions as a bridge between manager and rl-algo
@author: lehmann
"""
import rospy

from dqn_model import DQNModel
from rhbp_core.msg import ActivationState
from rhbp_core.srv import GetActivation, GetActivationResponse
import numpy


class RLComponentNode(object):
    def __init__(self):
        # gets name as a ros parameter
        self.name = rospy.get_param("~name", "rl_component_node")
        # True if the models is set up
        self.is_model_init = False
        # service for communicating the activations
        self._getStateService = rospy.Service(self.name + 'GetActivation', GetActivation,
                                              self._get_activation_state_callback)
        # choose appropriate model
        self.model = DQNModel(self.name)

        # save the last state
        self.last_state = None

        # default values of dimensions
        self.number_outputs = -1
        self.number_inputs = -1

        # current experience batch tuples (old_state,new_state,action,reward)
        self.reward_list = []

    def _get_activation_state_callback(self, request_msg):
        """
        answers the service and responds with the activations
        :param request_msg: GetActivation 
        :return: 
        """

        try:
            request = request_msg.input_state
            self.check_if_model_is_valid(request.num_inputs, request.num_outputs)
            # save current input state and update the model
            self.save_request(request)
            # update the last state
            self.last_state = request.input_state
            negative_states = request_msg.negative_states
            for state in negative_states:
                self.save_request(state)

            # transform the input state and get activation
            transformed_input = numpy.array(request.input_state).reshape(([1, len(request.input_state)]))
            activations = self.model.feed_forward(transformed_input)

            # return the activation via the service
            activations = activations.tolist()[0]
            activation_state = ActivationState(**{
                "name": self.name,  # this is sent for sanity check and planner status messages only
                "activations": activations,
            })
            return GetActivationResponse(activation_state)
        except Exception as e:
            rospy.logerr(e.message)
            return None

    def save_request(self, request):
        """
        save the old_state,new_state,action,reward tuple in a list for batch updating of the model
        :param request: 
        :return: 
        """
        if self.last_state is None:
            return
        last = numpy.array(self.last_state).reshape(([1, len(self.last_state)]))
        new = numpy.array(request.input_state).reshape(([1, len(request.input_state)]))
        reward_tuple = (last, new, request.last_action, request.reward)

        self.reward_list.append(reward_tuple)
        self.update_model()

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

    def update_model(self):
        """
        starts the training in the model for each tuple. 
        Note: the updating of the weights in the algorithm happens
        in a specified intervall
        :return: 
        """
        for element in self.reward_list:
            self.model.train_model(element)
        self.reward_list = []

    def init_model(self, num_inputs, num_outputs):
        """
        inits the model with the specified parameters
        :param num_inputs: 
        :param num_outputs: 
        :return: 
        """
        self.number_inputs = num_inputs

        self.number_outputs = num_outputs

        self.model.start_nn(num_inputs, num_outputs)

        self.reward_list = []

        self.is_model_init = True


if __name__ == '__main__':
    try:
        rospy.init_node('rl_node', anonymous=True)
        rl_component = RLComponentNode()

        rospy.spin()

    except rospy.ROSInterruptException:
        rospy.logerr("program interrupted before completion")
