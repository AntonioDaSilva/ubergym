import pickle
import numpy as np
import logging, sys
import gym

import drivers.RandomDriver as RandomDriver
from constants import kwargs_single_driver

# logging config
logging.basicConfig(
    format='%(message)s',
    stream=sys.stdout,
    level=logging.INFO,
)

# auxiliary function to get the number of passengers in each node
def get_passengers_by_node(env):
    p = np.zeros((len(env.map),))
    for passenger in env.passengers:
        p[passenger.position] += 1
    return p

# initialize environment
logging.info('Initializing environment')
num_steps = 10000
env = gym.make('ubergym/uber-v0', num_steps = num_steps, **kwargs_single_driver)

# reset and loop through environment
average_passengers = np.zeros((len(env.map),env.num_steps))
observation = env.reset()
done = False
step = 0

while not done:
    logging.info(f'Step: {step}')
    p = get_passengers_by_node(env)
    average_passengers[:,step] = p/(step+1)
    observation, rewards, done, info = env.step([0])
    step += 1

with open("./data/average_passengers.pkl", "wb") as f:
    pickle.dump(average_passengers, f)

with open("data/passenger_generation_probabilities.pkl", "wb") as f:
    pickle.dump(kwargs_single_driver["passenger_generation_probabilities"], f)
