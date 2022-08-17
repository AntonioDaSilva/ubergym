from envs.uber import Uber
import pickle
import numpy as np
import logging, sys
import matplotlib.pyplot as plt

import drivers.RandomDriver as RandomDriver

#TODO: pass only to each driver its own observation

# logging config
logging.basicConfig(
    format='%(message)s',
    stream=sys.stdout,
    level=logging.INFO,
)


#environment constants

n_drivers = 1

with open("generate_graph/graph.pkl", "rb") as f:
    G = pickle.load(f)

passenger_generation_probabilities = [0.1] * len(G)

# initialize environment

env = Uber(1, passenger_generation_probabilities, G)
driver = RandomDriver.Driver(0, len(G), True)

# reset and loop through environment

observations = env.reset()

done = False
step = 0

while not done:
    logging.info(f'Step: {step}')
    vectorized_observations = np.array(list(observations.values()))
    observation = vectorized_observations[:,0]
    action = driver.action(observation)
    driver.log()
    observations, rewards, done, info = env.step([action])
    driver.add_reward(rewards[0])
    driver.log()
    step += 1
    #done = True

plt.plot(np.arange(len(driver.rewards)), driver.rewards)
plt.show()

