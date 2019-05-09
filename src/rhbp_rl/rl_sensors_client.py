#!/usr/bin/env python2
"""
@author: gozman
"""
import rospy
from rhbp_rl.srv import RecSensor



def register_rl_sensor(name, include_in_rl=True, state_space=2, encoding=0):
    rospy.wait_for_service('RLRecSensor')
    reg_sensor = rospy.ServiceProxy('RLRecSensor', RecSensor)
    success = reg_sensor(name, include_in_rl, state_space, encoding)
    return success
