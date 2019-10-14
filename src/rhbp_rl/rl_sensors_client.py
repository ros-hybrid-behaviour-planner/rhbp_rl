#!/usr/bin/env python2
"""
@author: gozman
"""
import rospy
from rhbp_rl.srv import RecSensor, EpEnded, GetIndex, RegisterMarlGroup, AddToMarl, RemoveFromMarl



def register_rl_sensor(name, prefix='', include_in_rl=True, state_space=2, encoding=0):
    rospy.wait_for_service(prefix + '/RLRecSensor')
    reg_sensor = rospy.ServiceProxy(prefix + '/RLRecSensor', RecSensor)
    success = reg_sensor(name, include_in_rl, state_space, encoding)
    return success

def signal_episode_end(executed_behaviour, reward=0, prefix=""):
    rospy.wait_for_service(prefix + '/EpisodeEnd')
    ep_end = rospy.ServiceProxy(prefix + '/EpisodeEnd', EpEnded)
    success = ep_end(reward, executed_behaviour)
    return success

def get_b_index(executed_behaviour, service_name):
    rospy.wait_for_service(service_name)
    get_index = rospy.ServiceProxy(service_name, GetIndex)
    index = get_index(executed_behaviour)
    return index

def register_marl_group(prefix, *argv):
    """
    Register a group of marl agents via receving the manager names, these will 
    be used later to receive the actions of the corresponding agents
    :param prefix: this is the prefix of the manager which controlls the agent you want to register the group to
    :*argv: following arguments are names of the managers which are members of the group
    """
    service_name = prefix + "/register_marl_group"
    rospy.wait_for_service(service_name)
    names = []
    for name in argv:
        #if prefix != str(name):
        names.append(str(name))
    register = rospy.ServiceProxy(service_name, RegisterMarlGroup)
    success = register(names)
    return success

def add_to_marl_group(prefix, name):
    service_name = prefix + "/add_to_marl_group"
    rospy.wait_for_service(service_name)
    add = rospy.ServiceProxy(service_name, AddToMarl)
    success = add(name)
    return success

def remove_from_marl_group(prefix, name):
    service_name = prefix + "/remove_from_marl_group"
    rospy.wait_for_service(service_name)
    add = rospy.ServiceProxy(service_name, RemoveFromMarl)
    success = add(name)
    return success    
        

