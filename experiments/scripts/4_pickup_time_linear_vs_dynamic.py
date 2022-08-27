from lib2to3.pgen2 import driver
import pickle
import numpy as np
import logging, sys
import gym

from run import run

# logging config
logging.basicConfig(
    format='%(message)s',
    filename='logs.log',
    level=logging.INFO,
)

min_drivers = 2
max_drivers = 9
STEPS_PER_PASSENGER = 1 # we want 1 roughly one passenger per 1 step in expected value
n_episodes = 300
passenger_waiting_times = []
def callback(env):
    global passenger_waiting_times
    for passenger in env.passengers:
        if passenger.picked_up_at is not None:
            passenger_waiting_times.append(passenger.picked_up_at - passenger.spawned_at)

for i in range(min_drivers, max_drivers + 1):
    logging.info(f'Running with {i} drivers')
    passenger_waiting_times = []
    run(n_drivers = i, driver_type = 'Accepting', steps_per_passenger=STEPS_PER_PASSENGER, matcher_type='LINEAR_SUM', n_episodes=n_episodes, episode_callbacks=[callback])
    with open(f"data/linear_sum_passenger_waiting_times_{i}drivers.pkl", "wb") as f:
        pickle.dump(passenger_waiting_times, f)
    passenger_waiting_times = []
    run(n_drivers = i, driver_type = 'Accepting', steps_per_passenger=STEPS_PER_PASSENGER, matcher_type='DYNAMIC', n_episodes=n_episodes, episode_callbacks=[callback])
    with open(f"data/dynamic_passenger_waiting_times_{i}drivers.pkl", "wb") as f:
        pickle.dump(passenger_waiting_times, f) 