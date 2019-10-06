# rhbp_rl

This RHBP extensions adds an additional activation source based on
reinforcement learning. 

The implementation provides an alternative activation algorithm and can work with both single-agent und multi-agent scenarios. Multi-agent mode assissts agents in learning 
cooperatively by providing them with ability to track additional information about other agents
in the group. See further about the registration of marl groups. 


## Usage

In order to make use of this extension it is necessary to inject the alternative
activation algorithms as shown below:

```python
from behaviour_components.managers import Manager, ActivationAlgorithmFactory
from rhbp_rl.activation_algorithm import register_in_factory

register_in_factory(ActivationAlgorithmFactory)
m = Manager(activation_algorithm="reinforcement")

```

In order to register an rl sensor:

```python
from behaviour_components.managers import Manager, ActivationAlgorithmFactory
from rhbp_rl.activation_algorithm import register_in_factory
from rhbp_rl.rl_sensors_client import register_rl_sensor, signal_episode_end

register_in_factory(ActivationAlgorithmFactory)
m = Manager(activation_algorithm="reinforcement", prefix='my_prefix')
register_rl_sensor(name='boardHandSensor_0', prefix='my_prefix', include_in_rl=True, state_space=3, encoding=1)
#encoding 1 for one_hot encoding
```
In order to signal the end of the episode and save the last step:

```python
signal_episode_end(executed_behaviour=m.executed_behaviours[0].name,reward=2, prefix='my_prefix')
#Be aware that sometimes manager will not have exectuted behaviour with index 0, it is recommended to catch the execption (IndexError) and provide default behvaiour
```

In order to register MARL Group
```python
m = Manager(prefix='robot1', activation_algorithm="reinforcement", max_parallel_behaviours=1)
m2 = Manager(prefix='robot2', activation_algorithm="reinforcement", max_parallel_behaviours=1)
register_marl_group(prefix='robot1', 'robot2')
register_marl_group(prefix='robot2', 'robot1')
#the function first takes the prefix to the manager and any number of prefixes for other managers that are part of the marl group
```


## Default network 
The parameters that are relevant for the default neural network and ddql
```xml
    <param name="dropout_rate" type="double" value="0.0" />
    Dropout rate before the output layer, takes floatrs from 0.0 to 1.0
    <param name="timeseries" type="bool" value="false" />
    If timeseries mode is on, the training will be done in ordered batches. Important: If you want to use timeseries mode, you will need CUDA supporting GPU because the default variation NN implemented via tensorflow uses CUDNNLSTM cell as recurrent layer.
    <param name="timeseries_steps" type="int" value="8"/>
    The number of steps in one batch for timeseries training
    <param name="marl_steps" type="int" value="2"/>
    The number of past actions that are saved and observed by the agents in the marl group
    <param name="propagate" type="bool" value="true"/>
    Wether the reward for the last step should be propapgated to all steps in the episode or not,
    relevant only of the episodic mode is on
    <param name="episodic" type="bool" value="true"/>
    If the episodic mode is on, the transitions will be saved only after the episode has ended and it was signalled by the previous mentioned signal
    <param name="hard_update" type="bool" value="false" />
    If set to true, will update the target network by replacing it with the q-network instead of gradually updating it
    <param name="use_custom_network" type="bool" value="true"/>
    If set to true, rhbp will try to use the custom network
    <param name="sampling_rate" type="int" value="3"/>
    A parameter which determines how many time the memory buffer will be sampled during each training phase. 
    Generally, higher values yield better results when it is impossible to use large batches (upwards of hundreds of examples), which can be explained by the networks forgetfullness.
    <param name="hidden_layer_amount" type="int" value="3" />
    The number of layers that follow the input layer
    <param name="hidden_layer_cell_amount" type="int" value="20" />
    The number of units in each hidden layer
```

## Neural Network customization
The custom network should implement the interface provided in the rhbp_rl package. 
The network class should be put into rhbp_rl/src category and named RHBP_custom_network with class RHBPCustomNeuralNetwork in it. It will be imported using following code:

```python
 from RHBP_custom_network import RHBPCustomNeuralNetwork
```
RHBP_RL package has an example custom network implemented.

## Structure

###rhbp_rl
* **activation_algorithm.py** Extension of the BaseActivationAlgo considering activation through RL. Here, the calculation is gathered from the independent RL component and integrated into the overall activation calculation.
* **input_state_transformer.py** Transforms the input state of RHBP components to an abstract format used by the RL component.
* **exploration_strategies.py** Contains different possible exploration strategies. 
* **ddql.py** Contains an approximator agnostic implementation of DDQL algorithm
* **model_interface.py** Contains the intreface for the approximators which should be implemented in order to work with the DDQL algorithm implementation
* **expirience.py** An implementation of the expirience buffer for memory replay
* **neural_network.py** A default neural network 
* **rl_component.py:** ROS component that feeds the abstract input into the DQN algorithms and interprets the results.
* **rl_config.py:** Container classes for parameters of DQN and exploration.
* **rl_sensors_client.py** Containers various utilities for registrations of sensors, marl groups and so forth, details follow
* **RHBP_custom_network.py** An example of a custom neural network, change this file if you want to use a custom neural architecture


