import pickle
import numpy as np

# we want 1 roughly one passenger per 10 steps in expected value
STEPS_PER_PASSENGER = 10
SEED = 2002
np.random.seed(SEED)

with open("../generate_graph/graph.pkl", "rb") as f:
    G = pickle.load(f)

kwargs_single_driver = {
    'n_drivers': 1,
    'graph': G,
    'passenger_generation_probabilities': np.random.random(size = len(G))/(STEPS_PER_PASSENGER*len(G)) 
}