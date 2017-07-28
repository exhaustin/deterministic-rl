# Deterministic Reinforcement Learning
- Author: Austin Wang
- Year: 2017
- Requirements: Numpy (Matplotlib for visualization) (Tensorflow, Keras, for DDPG) (See requirements.txt for full details)


## Introduction
Collection of basic deterministic reinforcement learning agents and environments
Agents are also compatible with gym environments


## File list  
**learn-fp-ldpg.py**  
Applies off-policy deterministic actor-critic with linear models to the force-pose environment
Usage: `$python learn-fp-ldpg.py`  

**learn-fp-ddpg.py**  
Applies deep deterministic policy gradient to the force-pose environment
Usage: `$python learn-fp-ldpg.py`  

**learn-sp-ddpg.py**  

**nolearn-sp.py**  

**nolearn-fp.py**

**requirements.txt**
Python library list (output file of pip freeze)  

#### agents/  
Reinforcement learning agent modules  
Given any environment, an agent should be able to learn from normalized observations and rewards  

**LDPG_QL.py**  
Off-policy deterministic actor-critic with linear models

**DDPG_QL.py**  
Off-policy deterministic actor-critic with neural models (Deep Deterministic Policy Gradient)  

**DQN.py**  
Q-learning with neural models (Deep Q-Learning)  

**DDPG_SARSA.py**  

**offline_DDPG_SARSA.py**

**models/**  
Contains model files

**misc/**  
Miscellaneous modules such as replay buffer

#### envs/  
Virtual environment modules for testing learning algorithms  
Maintains system states, allows action input, returns observations and rewards  

**ForcePose.py**  
Simulation of an object orientation problem  
- System states X - position and orientation (Pose)
- Observations F - forces and torques
- Actions dX - change of position and orientation
- System: F = K(X - X_op)		(K and X_op are randomly initialized)
- System dynamics: X[n+1] = X[n] + dX[n]

