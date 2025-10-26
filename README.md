# MPC Multi-Agent Rollout

This project contains files of Model Predictive Control (MPC) with multi-agent rollout capabilities for reinforcement learning environments.

## Installation

### 1: Install PettingZoo with MPE

First, install PettingZoo with the Multi-Particle Environment (MPE) suite:

```bash
pip install 'pettingzoo[mpe]'
```

### 2: PettingZoo Documentation

For detailed information about PettingZoo environments and their implementations, please refer to the official documentation:

- **PettingZoo Documentation**: https://pettingzoo.farama.org/
- **MPE Environments**: https://pettingzoo.farama.org/environments/mpe/

### 3: Modify Environment and Reward Functions

This project requires modifications to the base PettingZoo MPE environment files to customize the environment dynamics and reward functions.

#### Modified Files Location

The modified environment files can be found in the repository at:

```
https://github.com/Athmajan/mpc_multi_agent_rollout/tree/adversaryfix/env_modified_files
```

This folder contains:
- **Original files**: The base PettingZoo environment files
- **Modified files**: The customized versions with updated reward functions and environment modifications

Replace the original files with the modifications.


### 4: Base policy implementation

The file MPE_Base.py contains implementation of base policy to pick the closest target and move towards it.


### 5: MPPI rollout implementation

The file mppi_mpe.py contains implementation of base policy to pick the closest target and move towards it.

### 6 : World model implementation
Use world_model.py to train a world model of the MPE environment using TD

### 7 : Collect saved sequential rollout data to train the communication signal of an agent
modelAgent0/getExpReplay.py collects saved data and pre process for training the model of the communication signal

### 8 : Train the model of an agent
modelAgent0/modelA0.py trains the model.
