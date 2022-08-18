import pickle
import numpy as np
import logging, sys
import gym

from constants import kwargs_single_driver
import drivers.QLearningDriver as QLearningDriver

# logging config
logging.basicConfig(
    format='%(message)s',
    stream=sys.stdout,
    level=logging.INFO,
)


env = gym.make('ubergym/uber-v0', is_logging = False,   **kwargs_single_driver)
G = kwargs_single_driver["graph"]
n_drivers = kwargs_single_driver["n_drivers"]
drivers = [QLearningDriver.Driver(i, len(G), G, True) for i in range(n_drivers)]

# reset and loop through environment
num_episodes = 100

for episode in range(num_episodes):
    logging.info(f'Epsiode: {episode}')
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
        with open(f"data/qlearning_driver{i}_episode_{episode}_rewards.pkl", "wb") as f:
            pickle.dump(rewards, f)

    for driver in drivers:
        driver.reset()