"""
transforms values from rhbp to rl-values
@author: lehmann, hrabia
"""
import numpy
from behaviour_components.sensors import EncodingConstants
from rl_config import TransitionConfig
import rospy
from rhbp_rl.srv import RecSensor

import utils.rhbp_logging
rhbplog = utils.rhbp_logging.LogManager(logger_name=utils.rhbp_logging.LOGGER_DEFAULT_NAME + '.rl')

class RlExtension(object):
    """
    This Extension can be included in the Sensors. It determines how the true values of the sensors should be used the
    RL-algorithm.
    # Encoding types = [ hot_state , none] see EncodingConstants above
    """
    # TODO state_space does not work for negative numbers!

    def __init__(self, name, encoding=0, state_space=2, include_in_rl=True):
        self.name = name
        self.encoding = encoding
        # State space is only required for EncodingConstants.HOT_STATE
        self.state_space = state_space
        self.include_in_rl = include_in_rl  # True if it should be used for learning
class GoalInfo(object):
    """
    Simple helper class we use to temporarily store goal information we need for reward calculation
    """

    def __init__(self, name, fulfillment, is_permanent, priority, satisfaction_threshold):
        self.name = name
        self.fulfillment = fulfillment
        self.is_permanent = is_permanent
        self.priority = priority
        self.satisfaction_threshold = satisfaction_threshold


class InputStateTransformer(object):
    """
    this class gets called in the activation algorithm and transform the rhbp components into the InputStateMessage
    """

    def __init__(self, manager):
        self._manager = manager
        self.conf = TransitionConfig()
        self._last_operational_goals = {}
        self.sensor_descriptors = {}
        self._set_rl_descriptor_service = rospy.Service('RLRecSensor', RecSensor, self._set_rl_descriptor_callback)

    def _set_rl_descriptor_callback(self, req):
        rl_ext = RlExtension(
            name=req.name, state_space=req.state_space, encoding=req.encoding, include_in_rl=req.include_in_rl)
        self.sensor_descriptors[req.name] = rl_ext
        return 1


    def calculate_reward(self):
        """
        this function calculates regarding the fulfillment and priorities of the active goals
        a reward value. 
        :return: reward value
        """

        reward_value = 0

        # we collect the information we need about the goals in a more slim data structure
        current_operational_goals = {g.name: GoalInfo(g.name, g.fulfillment, g.isPermanent, g.priority,
                                                      g.satisfaction_threshold)
                                     for g in self._manager.operational_goals}

        # we are using goal.priority+1, because the default priority is 0.

        # first we check the difference starting from former registered goals
        for name, g in self._last_operational_goals.iteritems():  # check goals that have been operational before
            # goal is not anymore listed in operational goals.
            if name not in current_operational_goals:
                # collect reward from completed achievement (non-permanent) goals
                if not g.is_permanent:
                    # here we just use the satisfaction threshold by default 1
                    reward_value += g.satisfaction_threshold * (g.priority+1)
            else:  # if it is still operational we compare the difference of fulfillment (current-last)
                fulfillment_delta = current_operational_goals[g.name].fulfillment - g.fulfillment
                reward_value += fulfillment_delta * (g.priority+1)

        # next we have to calculate the reward for all goals that have not yet been registered in the former step
        for name, goal in current_operational_goals.iteritems():
            if name not in self._last_operational_goals:
                if goal.satisfaction_threshold < goal.fulfillment:
                    fulfillment_delta = goal.satisfaction_threshold
                else:
                    fulfillment_delta = goal.fulfillment
                reward_value += fulfillment_delta * (goal.priority + 1)
                # the else case was already addressed in the loop above

        self._last_operational_goals = current_operational_goals

        return reward_value

    def behaviour_to_index(self, name):
        """
        gives for a given name of a behavior name the index in the behavior list
        :param name: 
        :return: 
        """
        num = 0
        for b in self._manager.behaviours:
            if b == name:
                return num
            num += 1
        return None

    def make_hot_state_encoding(self, state, num_state_space):
        """
        encodes the variables into a hot state format.
        :param state: 
        :param num_state_space: 
        :return: 
        """
        state = int(state)
        return numpy.identity(num_state_space)[state:state + 1].reshape([num_state_space, 1])

    def transform_input_values(self):
        """
        this function uses the wishes and sensors to create the input vectors
        :return: input vector
        """
        # init input array with first row of zeros
        input_array = numpy.zeros([1, 1])
        # extend input array with the sensors from conditions/behaviours
        input_array, sensor_input = self.transform_behaviours(input_array)
        # extend input array with the sensors from goals
        input_array = self.transform_goals(sensor_input, input_array)
        # cut first dummy line
        input_array = input_array[1:]
        return input_array

    def transform_behaviours(self, input_array):
        """
        extend the input array with the sensor values and wishes from the behaviours
        :param input_array: the input aray to be extended
        :return: the extended input array
        """
        use_wishes = self.conf.use_wishes
        use_true_value = self.conf.use_true_values
        # extend array with input vector from wishes
        sensor_input = {}
        # get sensor values from conditions via the behaviours
        for behaviour in self._manager.behaviours:
            # check for each sensor in the goal wishes for behaviours that have sensor effect correlations
            if use_wishes:
                for wish in behaviour.wishes:
                    wish_row = numpy.array([wish.indicator]).reshape([1, 1])
                    input_array = numpy.concatenate([input_array, wish_row])
            for sensor_value in behaviour.sensor_values:
                if not sensor_input.has_key(sensor_value.name) and use_true_value:
                    if sensor_value.name in self.sensor_descriptors:
                        #print('Will we skip? ' + sensor_value.name + ' ' + str(self.sensor_descriptors[sensor_value.name].include_in_rl))
                        if self.sensor_descriptors[sensor_value.name].include_in_rl == False: #The sensor should be explicitly turned off if it should be used for RL
                            continue
                        if self.sensor_descriptors[sensor_value.name].encoding == 1:
                            value = self.make_hot_state_encoding(sensor_value.value, self.sensor_descriptors[sensor_value.name].state_space)
                    else:
                        value = numpy.array([[sensor_value.value]])
                    sensor_input[sensor_value.name] = value
                    input_array = numpy.concatenate([input_array, value])
        return input_array, sensor_input

   

    def transform_goals(self, sensor_input, input_array):
        """
        transform only the goals
        :param sensor_input: saves which sensors were already included
        :param input_array: the inputs from the behaviour sensors
        :return: the updated input array. includes now the sensors of goals
        """
        use_wishes = self.conf.use_wishes
        use_true_value = self.conf.use_true_values
        # get sensors from goals
        for goal in self._manager.goals:
            for sensor_value in goal.sensor_values:
                if not sensor_input.has_key(sensor_value.name) and use_true_value:
                    if sensor_value.name in self.sensor_descriptors:
                        if self.sensor_descriptors[sensor_value.name].include_in_rl == False: #The sensor should be explicitly turned off if it should not be used for RL
                            continue
                        if self.sensor_descriptors[sensor_value.name].encoding == 1:
                            value = self.make_hot_state_encoding(sensor_value.value, self.sensor_descriptors[sensor_value.name].state_space)
                    else:
                        value = numpy.array([[sensor_value.value]])
                    sensor_input[sensor_value.name] = value
                    input_array = numpy.concatenate([input_array, value])
            # include wishes
            if use_wishes:
                for wish in goal.wishes:
                    wish_row = numpy.array([wish.indicator]).reshape([1, 1])
                    input_array = numpy.concatenate([input_array, wish_row])

        return input_array
