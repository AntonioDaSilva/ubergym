import pickle
from typing import Callable, List
import numpy as np
import logging, sys
import gym

import drivers.AcceptingDriver as AcceptingDriver
import drivers.RandomDriver as RandomDriver

# logging config
logging.basicConfig(
    format='%(message)s',
    filename='logs.log',
    level=logging.INFO,
)


def run(n_drivers, driver_type, steps_per_passenger, matcher_type, n_episodes, driver_logging = False, simulation_logging = False, episode_callbacks: List[Callable] = []):

    with open("../generate_graph/graph.pkl", "rb") as f:
        G = pickle.load(f)
    passenger_generation_probabilities = np.random.random(size = len(G))/(steps_per_passenger*len(G)) 

    env = gym.make('ubergym/uber-v0', 
        n_drivers = n_drivers, 
        passenger_generation_probabilities = passenger_generation_probabilities,
        graph = G,
        matcher_type = matcher_type,
        is_logging = simulation_logging)

    if driver_type == 'Accepting':
        drivers = [AcceptingDriver.Driver(i, len(G), G, driver_logging) for i in range(n_drivers)]
    elif driver_type == 'Random':
        drivers = [RandomDriver.Driver(i, len(G), G, driver_logging) for i in range(n_drivers)]

    for episode in range(n_episodes):
        # logging.info(f'Episode: {episode}')
        # reset and loop through environment
        observations = env.reset()
        done = False
        step = 0

        while not done:
            #logging.info(f'Step: {step}')
            vectorized_observations = np.array(list(observations.values()))
            actions = [drivers[i].action(vectorized_observations[:,i]) for i in range(n_drivers)]
            for driver in drivers:
                driver.log()
            observations, rewards, done, info = env.step(actions)
            for i in range(n_drivers):
                drivers[i].add_reward(rewards[i])
                drivers[i].log()
            step += 1
        
        for f in episode_callbacks:
            f(env)

    return
        



    

