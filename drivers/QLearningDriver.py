from typing import Iterable, List, Tuple
from dataclasses import dataclass
import logging, sys
import numpy as np
import networkx as nx

from ubergym.envs.maps import Map

# logging config
logging.basicConfig(
    format='%(message)s',
    stream=sys.stdout,
    level=logging.INFO,
)

@dataclass
class Driver:
    name: int
    num_actions: int
    graph: nx.DiGraph 
    is_logging: bool = False
    lr: float = 0.05
    epsilon: float = 0.5


    def __post_init__(self):
        self.rewards = []
        self.actions = []
        self.observations = []
        self.states = []
        self.messages = []
        self.state_to_message = {
            0: "IDLE",
            1: "MATCHING", 
            2: "MATCHED", 
            3: "RIDING", 
            4: "OFF"
        }
        self.map = Map(self.graph)

        # initalize RL values
        self.inital_low = 0
        self.inital_high = 50
        self.discount = 1.0
        self.num_price_states = 20
        self.max_price = 200
        self.n_nodes = len(self.graph)
        print(self.n_nodes)
        self.qtable = [
            # IDLE: driver position, passenger position, action
            np.random.uniform(self.inital_low, self.inital_high, size = (self.n_nodes, self.num_actions)), 
            # MATCHING: driver position, passenger destination, passenger position, price, action
            np.random.uniform(self.inital_low, self.inital_high, size = (self.n_nodes, self.n_nodes, self.n_nodes, self.num_price_states, 2)), 
            # MATCHED: driver position, passenger destination, passenger position, action
            np.random.uniform(self.inital_low, self.inital_high, size = (self.n_nodes, self.n_nodes, self.n_nodes, self.num_actions)),
            # RIDING: driver position, passenger destination, action
            np.random.uniform(self.inital_low, self.inital_high, size = (self.n_nodes, self.n_nodes, self.num_actions))
        ]

        self.prev_state = None
        self.prev_action = None

        # initalize non-neighbor movements to -np.inf so that they are not chosen
        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                if i == j:
                    continue
                if self.map.neighbors(i,j):
                    continue
                self.qtable[0][i][j] = -np.inf
        
        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                for k in range(self.n_nodes):
                    if i == k:
                        continue
                    if self.map.neighbors(i,k):
                        continue
                    self.qtable[3][i][j][k] = -np.inf
                    self.qtable[2][i][j][k] = -np.inf


    def action(self, observation) -> int:
        state = self._get_state(observation)
        self._learn(state)

        self.observations.append(observation)
        self.states.append(state)
        self._log_observation(observation)
        
        action = self._select_action(state)

        return action

    def _learn(self, state):
        if self.prev_state is None:
            self.prev_state = state
            return
        
        # TD Learning
        reward = self.rewards[-1]
        qmax = np.max(self.qtable[state[0]][state[1:]])
        qcurr = self.qtable[self.prev_state[0]][self.prev_state[1:]][self.prev_action]
        self.qtable[self.prev_state[0]][self.prev_state[1:]][self.prev_action] += self.lr * (reward + self.discount * qmax - qcurr)
        self.prev_state = state

    def reset(self):
        self.rewards = []
        self.actions = []
        self.observations = []
        self.states = []
        self.messages = []
        self.prev_state = None
        self.prev_action = None

    def _get_state(self, observation) -> Tuple:
        state = tuple(observation.astype(int))
        if state[0] == 0:
            return state[:2]
        elif state[0] == 1:
            return *state[:4], self._get_price_state(observation[4])
        elif state[0] == 2:
            return state[:4]
        elif state[0] == 3:
            return state[:3]
        return state

    def _get_next_node(self, position, destination) -> int:
        if position == destination:
            return position
        else:
            path = self.map.shortest_path(position, destination)
            nxt = path[1]
            return nxt

    def _select_action(self, state):

        if state[0] == 2:
            position = state[1]
            destination = state[3]
            action = self._get_next_node(position, destination)
        elif state[0] == 3:
            position = state[1]
            destination = state[2]
            action = self._get_next_node(position, destination)

        elif np.random.random() < self.epsilon:
            action = np.random.randint(0, len(self.qtable[state[0]][state[1:]]))
        else:
            action = np.argmax(self.qtable[state[0]][state[1:]])
        self.prev_action = action
        self._log_action(action)
        self.actions.append(action)
        return action

    def _get_price_state(self, price: float) -> int:
        if price >= self.max_price:
            return self.num_price_states - 1
        elif price < 0:
            return 0
        else:
            return int(price // int(self.max_price / self.num_price_states))


    def add_reward(self, reward):
        self._log_reward(reward)
        self.rewards.append(reward)

    def log(self):
        if not self.is_logging:
            return
        
        for message in self.messages:
            logging.info(message)
        
        self.messages = []

    def _log_action(self, action):
        self.messages.append(f'Driver {self.name}, Action = {action}')

    def _log_reward(self, reward):
        self.messages.append(f'Driver {self.name}, Reward = {reward}')

    def _log_observation(self, observation):
        driver_msg = f'Driver {self.name}'
        state = self.state_to_message[observation[0]]
        state_msg = f'State = {state}'
        position_msg = f'Position = {observation[1]}'
        p_destination_msg = f'Passenger Destination = {observation[2]}'
        p_position_msg = f'Passenger Position = {observation[3]}'
        price_message = f'Price = {observation[4]}'

        message = driver_msg + ', ' + state_msg + ', ' + position_msg
        
        if state == "IDLE" or state == 'OFF':
            self.messages.append(message)
        elif state == "MATCHING":
            message += ', ' + p_destination_msg + ', ' + p_position_msg + ', ' + price_message
            self.messages.append(message)
        elif state == "MATCHED":
            message += ', ' + p_destination_msg + ', ' + p_position_msg
            self.messages.append(message)
        elif state == "RIDING":
            message += ', ' + p_destination_msg
            self.messages.append(message)

        return 