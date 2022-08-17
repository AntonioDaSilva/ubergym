import pickle
import numpy as np
import logging, sys
import gym

from constants import kwargs_single_driver
import drivers.AcceptingDriver as AcceptingDriver

# logging config
logging.basicConfig(
    format='%(message)s',
    stream=sys.stdout,
    level=logging.INFO,
)


env = gym.make('ubergym/uber-v0', **kwargs_single_driver)
G = kwargs_single_driver["graph"]
n_drivers = kwargs_single_driver["n_drivers"]
drivers = [AcceptingDriver.Driver(i, len(G), G, True) for i in range(n_drivers)]

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

for i in range(n_drivers):
    rewards = drivers[i].rewards
    with open(f"data/accepting_driver{i}_rewards.pkl", "wb") as f:
        pickle.dump(rewards, f)