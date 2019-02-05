#! /usr/bin/env python2
'''
Tests the goal implementations

Created on 29.01.2019

@author: hrabia
'''

import time
import unittest

import rospy
import rostest

from behaviour_components.behaviours import BehaviourBase
from behaviour_components.activators import BooleanActivator
from behaviour_components.conditions import Condition
from behaviour_components.condition_elements import Effect
from behaviour_components.goals import GoalBase
from behaviour_components.managers import Manager, ActivationAlgorithmFactory
from behaviour_components.sensors import Sensor

from rhbp_rl.activation_algorithm import register_in_factory

PKG = 'rhbp_rl'


"""
System test for RL component. Assumes, that a rosmaster is running
"""


class BehaviourA(BehaviourBase):

    def do_step(self):
        rospy.logwarn("A")


class BehaviourB(BehaviourBase):

    def do_step(self):
        rospy.logwarn("B")


class TestReinforcementLearning(unittest.TestCase):
    """
    Basic RL component test
    """

    def __init__(self, *args, **kwargs):
        super(TestReinforcementLearning, self).__init__(*args, **kwargs)
        # prevent influence of previous tests
        self.__message_prefix = 'TestRL' + str(time.time()).replace('.', '')
        rospy.init_node('rl_test_node', log_level=rospy.DEBUG)
        # Disable planner, since the change from python to C
        #  disturbs the connection between the test process and the node process
        rospy.set_param("~planBias", 0.0)

    def test_rl_basic(self):

        method_prefix = self.__message_prefix + "test_rl_basic"
        planner_prefix = method_prefix + "Manager"
        register_in_factory(ActivationAlgorithmFactory)
        m = Manager(activationThreshold=7, prefix=planner_prefix, activation_algorithm="reinforcement")

        behaviour_a = BehaviourA(name=method_prefix+"A", planner_prefix=planner_prefix)
        behaviour_b = BehaviourB(name=method_prefix+"B", planner_prefix=planner_prefix)

        sensor = Sensor(name="bool_sensor", initial_value=False)

        # same effect so planner cannot distinguish
        behaviour_a.add_effect(Effect(sensor_name=sensor.name, indicator=1.0))
        behaviour_b.add_effect(Effect(sensor_name=sensor.name, indicator=1.0))

        goal = GoalBase(method_prefix + '_goal', planner_prefix=planner_prefix)
        goal.add_condition(Condition(sensor, BooleanActivator()))

        for x in range(0, 10, 1):
            m.step()
            rospy.sleep(0.1)

        # TODO set result after some steps.

        # TODO evaluate
        # goal_proxy = m.goals[0]
        # goal_proxy.fetchStatus(3)
        # self.assertTrue(goal_proxy.satisfied, 'Goal is not satisfied')


if __name__ == '__main__':
    rostest.rosrun(PKG, 'test_rl_node', TestReinforcementLearning)
    rospy.spin()
