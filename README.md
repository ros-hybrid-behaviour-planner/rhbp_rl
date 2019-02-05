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