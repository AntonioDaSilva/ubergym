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

n_drivers = 5
STEPS_PER_PASSENGER = np.linspace(.6, 1.2, 13)
n_episodes = 300
passenger_waiting_times = []
def callback(env):
    global passenger_waiting_times
    for passenger in env.passengers:
        if passenger.picked_up_at is not None:
            passenger_waiting_times.append(passenger.picked_up_at - passenger.spawned_at)

for i in range(len(STEPS_PER_PASSENGER)):
    steps_per_passenger = STEPS_PER_PASSENGER[i]
    logging.info(f'Running with {steps_per_passenger = }')
    passenger_waiting_times = []
    run(n_drivers = n_drivers, driver_type = 'Accepting', steps_per_passenger=steps_per_passenger, matcher_type='LINEAR_SUM', n_episodes=n_episodes, episode_callbacks=[callback])
    with open(f"data/linear_sum_passenger_waiting_times_{n_drivers}drivers_{steps_per_passenger}stepsperpassenger.pkl", "wb") as f:
        pickle.dump(passenger_waiting_times, f)
    passenger_waiting_times = []
    run(n_drivers = n_drivers, driver_type = 'Accepting', steps_per_passenger=steps_per_passenger, matcher_type='DYNAMIC', n_episodes=n_episodes, episode_callbacks=[callback])
    with open(f"data/dynamic_passenger_waiting_times_{n_drivers}drivers_{steps_per_passenger}stepsperpassenger.pkl", "wb") as f:
        pickle.dump(passenger_waiting_times, f) 