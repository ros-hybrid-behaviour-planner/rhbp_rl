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


###Ideas for changes in basic RL architecture and codebase:
- Refactor some things, see comments, todos and suggestions in code 
- (Not sure about that)Create a database module which will work with Input-transformer to provide a storage and history which can also be dumped into file if needed, this module will supply the training examples to neural network algorithm
- Change nn_model_base to Abstract Approximator, make it an actual abstact class, see next points
- Write an interface for DQN that will allow to use custom NNs for Q-Learning and training

For the genericness and higher flexibility with the application for deep learning to reinforcement learning the access to the deep model should be completely generic and not dependent on the actual implementation of the model. This includes usage of the different frameworks for deep learning. This creates a design question of what kind of interface can be created in order to completely abstract the usage of model my the agent. There is a number of functionale that is needed irregardless of specific architecture:
- Prediction: this the main function, getting the infered values from the neural network.
- Training: training is needed to increase the quality of predictions. The question, however, stands whether the database (training samples) should be completely decoupled from the model or not. Considering the genericness of the samples and independence of the data points from the neural network model (inputs are always the same, predicated on the task RHBP is trying to solve), it would be better to indeed abstact the database from network. Training method with then take the full example in order to run backpropagation and learn. The deep learning necessitates training on batches instead of single examples, this problem can be solved by providing a batch length as a configuration property to the rl controller
- Saving model
- Loading model
GOAL OF THIS IS TO MAKE IT POSSIBLE TO USE DIFFERENT NN FRAMEWORKS
- Transition to tensorflow-2.0.0
- Abstact RL-Activation algo into RL controller and activation algo (MAKE THIS MORE SPECIFIC, maybe no new abstractions are needed and we just need to juggle some functionale) THIS MIGHT HELP IN CASES WHERE WE WANT TO GET MORE DIFFERNET RL ALGOS IN THE FUTURE
- DQN class is now both model and algo, modularise them a bit further into RL ALGO and NN
- Refactor activation algorithm in a more functional way (currently uses quite a lof of global variables, hard to follow in a top down approach)
SO THE GOAL WOULD BE TO REFACTOR AND COME TO THE SAME FUNCTIONALE THAT WE CURRENTLY HAVE
- Exploration strategies: creatge a hierarchy of classes as suggested, create an interface and modularise