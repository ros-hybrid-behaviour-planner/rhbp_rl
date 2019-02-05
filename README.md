# rhbp_rl

This RHBP extensions adds an additional activation source based on
reinforcement learning. 

The implementation provides an alternative activation algorithm.

The realised DDQN approach is applying tensorflow.


## Usage

In order to make use of this extension it is necessary to inject the alternative
activation algorithms as shown below:

```python
from behaviour_components.managers import Manager, ActivationAlgorithmFactory
from rhbp_rl.activation_algorithm import register_in_factory

register_in_factory(ActivationAlgorithmFactory)
m = Manager(activation_algorithm="reinforcement")

```

## Structure

### rhbp_core

* **behaviour_components.sensors:** Contains RLExtension to add additional required configuration and information about the sensors, which is used for learning. 

Here, it would be the goal to refactor this in a way that rhbp_core does not contain any RL dependencies.

###rhbp_rl
* **activation_algorithm.py:** Extension of the BaseActivationAlgo considering activation through RL. Here, the calculation is gathered from the independent RL component and integrated into the overall activation calculation.
* **input_state_transformer.py:** Transforms the input state of RHBP components to an abstract format used by the RL component.
* **exploration_strategies.py:** Contains different possible exploration strategies. 
* **nn_model_base.py:**	Basic functions of the RLAlgorithmus.
* **dqn_model.py:** Actual DQN implementation. (Q-Network, ExperienceBuffer, DDQN)
* **rl_component.py:** ROS component that feeds the abstract input into the DQN algorithms and interprets the results.
* **rl_config.py:** Container classes for parameters of DQN and exploration.
