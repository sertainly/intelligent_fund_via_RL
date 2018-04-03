# A model of intelligent behavior in agent-based financial systems

This repository contains the code used in the paper I wrote as contribution to a research seminar as part of my Master in Banking and Finance Program at the Uni of St. Gallen. 

__Abstract:__

We propose a novel approach to model the behavior of institutions in agent-based financial systems. To achieve this, reinforcement learning techniques are used to continuously improve the demand decision of a fund during episodes of interaction with other agents in a simulated environment. It is shown that after training, the learning agent displays a sophisticated demand behavior. Also, it is not able to learn a dominant strategy. Lastly, the learning fund does not follow the dynamics of other, simpler agents, but displays more complex behavior.
The results indicate a potential for similar methods in the analysis of macroprudential stress scenarios.

## Thurner Model

Is is an implementation of the model described by Thurner, Farmer & Geanakoplos:
[Leverage causes fat tails and clustered volatility, 2012](https://arxiv.org/abs/0908.1555).

#### [thurner_model.py](thurner_model.py)

Contains the main classes for agents which populate the environment, as well as the mechanism to determine the equilibrium price in the market.

I strongly rely on Luzius Meisser's well documented implementation of the model, which was written for the Santa Fe Institute Complexity Economics MOOC and can be found [here](https://github.com/kronrod/sfi-complexity-mooc/blob/master/notebooks/leverage.ipynb). 

#### [env_thurner_model.ipynb](env_thurner_model.ipynb)

Here, the basic environment described by [Thurner et al.](https://arxiv.org/abs/0908.1555) is implemented similar to an [OpenAI gym](https://gym.openai.com) environment. It uses the classes in [thurner_model.py](thurner_model.py). We can check the functionality by running simulations and compare the results to the ones obtained by Thurner et al.

## Learning Fund

#### [learning_fund.py](learning_fund.py)

This is the heart of the model. Here, a new `LearningFund` class is introduced, which does not have a static demand function, but learns it via the actor-critic method.

The code for the actor-critic learning mechanism is based on an implementation by Denny Britz, which he uses to solve the
Continuous Mountain Car problem. It can be found [here](https://github.com/dennybritz/reinforcement-learning/blob/master/PolicyGradient/Continuous%20MountainCar%20Actor%20Critic%20Solution.ipynb).

I make adjustments to his code to make it compatible with the model described by Thurner et al. 

The environment now consists of the basic agents and mechanisms described by Thurner et al., including 10 normal `Funds` with static demand functions. Also we add one additional `Learning Fund`.

To run the simulation and let the `Learning Fund` learn:
```
python learning_fund.py experiment_name
```

The default number of episodes and time steps per episodes are set to 30 and 5000 respectively, and can be specified like so:
```
python learning_fund.py experiment_name -ep 30 -ts 5000
```

## Evaluations

#### [visualize_learning.ipynb](visualize_learning.ipynb)

Contains different visualizations of one long learning period (100 episodes).

#### [simulate_with_trained_fund.ipynb](simulate_with_trained_fund.ipynb)

Uses a pre-trained `LearningFund` (which learned for 100 episodes) and lets us run different kinds of scenarios with it.
