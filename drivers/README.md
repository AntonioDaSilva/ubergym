# Drivers

This folder contains example drivers to act on the simulation.

## RandomDriver

This driver chooses a random action regardless of the state.

## AcceptingDriver

This driver accepts all match requests regardless of the price and takes the shortest path to the destination.

## QLearningDriver

This driver uses a simple Q-table in order to decide its actions and learns by interacting with the environment. Very slow convergence due to the fact that the environment has very sparse rewards.

## MonteCarloDriver (TO BE IMPLEMENTED)

This driver accounts for the sparse rewards in the environment and makes updates at the end of the episode after collecting some data. This will make sure that the updates are meaningful compared to TD(0) method in which most of the updates are meaningless since the driver doesn't receive a reward until it drops off the passenger and gets paid.
