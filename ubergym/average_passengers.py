from envs.uber import Uber
import pickle
import numpy as np
import matplotlib.pyplot as plt

n_drivers = 1

def get_passengers_by_node(env):
    p = np.zeros((len(env.map),))
    for passenger in env.passengers:
        p[passenger.position] += 1
    return p

with open("generate_graph/graph.pkl", "rb") as f:
    G = pickle.load(f)

passenger_generation_probabilities = [0.1] * len(G)

env = Uber(1, passenger_generation_probabilities, G)
observation = env.reset()

done = False

average_passengers = np.zeros((len(env.map),env.num_steps))
step = 0
while not done:
    print(f'Step: {step}', end = '\r')
    p = get_passengers_by_node(env)
    average_passengers[:,step] = p/(step+1)
    observation, rewards, done, info = env.step([0])
    step += 1

print(observation)
print(rewards)
print(done)
print(info)

plt.title("Average passengers per node")
plt.xlabel("Step")
plt.ylabel("Passengers")


for i in range(len(G)):
    pi = average_passengers[i,:]
    plt.plot(np.arange(len(pi)), pi, label = str(i))

plt.legend()
plt.show()
