import pickle
import numpy as np
import logging, sys

import drivers.RandomDriver as RandomDriver
import gym


# auxiliary function to get the number of passengers in each node
def get_passengers_by_node(env):
    p = np.zeros((len(env.map),))
    for passenger in env.passengers:
        p[passenger.position] += 1
    return p

# initialize environment
n_drivers = 1
with open("../generate_graph/graph.pkl", "rb") as f:
    G = pickle.load(f)
passenger_generation_probabilities = [0.1] * len(G)

env = gym.make('ubergym/uber-v0', n_drivers = n_drivers, passenger_generation_probabilities = passenger_generation_probabilities, graph = G)

# reset and loop through environment
average_passengers = np.zeros((len(env.map),env.num_steps))
observation = env.reset()
done = False
step = 0

while not done:
    print(f'Step: {step}', end = '\r')
    p = get_passengers_by_node(env)
    average_passengers[:,step] = p/(step+1)
    observation, rewards, done, info = env.step(env.action_space.sample())
    step += 1

with open("./data/average_passengers.pkl", "wb") as f:
    pickle.dump(average_passengers, f)
