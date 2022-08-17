import pickle
import numpy as np
import logging, sys
import matplotlib.pyplot as plt

import drivers.RandomDriver as RandomDriver
import gym

# logging config
logging.basicConfig(
    format='%(message)s',
    stream=sys.stdout,
    level=logging.INFO,
)


#environment constants

n_drivers = 1
with open("../generate_graph/graph.pkl", "rb") as f:
    G = pickle.load(f)
passenger_generation_probabilities = [0.1] * len(G)

# initialize environment
env = gym.make('ubergym/uber-v0', n_drivers = n_drivers, passenger_generation_probabilities = passenger_generation_probabilities, graph = G)
drivers = [RandomDriver.Driver(0, len(G), G, True)]

# reset and loop through environment
observations = env.reset()

done = False
step = 0

while not done:
    logging.info(f'Step: {step}')
    vectorized_observations = np.array(list(observations.values()))
    actions = [drivers[i].action(vectorized_observations[:,i]) for i in range(n_drivers)]
    for driver in drivers:
        driver.log()
    observations, rewards, done, info = env.step(actions)
    for i in range(n_drivers):
        drivers[i].add_reward(rewards[i])
        drivers[i].log()
    step += 1

rewards = drivers[0].rewards

with open("./data/random_driver_rewards.pkl", "wb") as f:
    pickle.dump(rewards, f)