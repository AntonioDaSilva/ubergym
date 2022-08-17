# Ubergym

This repository contains an [OpenAI gym](https://gym.openai.com/) environment to simulate drivers in a Uber-like environment. The main purpose of this environment is to test multi-agent reinforcement learning algorithms for equilibrium computation in a complex setting, especially through changing rewards in order to create special game classes under which there are theoretical guarantees of convergence to various types of equilibriums. Furthermore, one can also change the environment dynamics by writing their own 'Matcher' object and changing passenger generation dynamics, which allows for testing of meta/transfer learning algorithms for agents to adapt in environments with same state space, however, with different dynamics.

# Setup

We first recommend creating a new conda environment with Python 3.9+:

```bash
conda create -n ubergym python=3.9
conda activate ubergym
```

Then, you can clone this repository and install the environment as a package:

```bash
git clone https://github.com/AntonioDaSilva/ubergym.git
cd ubergym
pip3 install -e .
```
