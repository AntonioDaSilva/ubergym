from typing import Iterable, List, Tuple
from dataclasses import dataclass
import logging, sys
import numpy as np
import networkx as nx

from ubergym.envs.maps import Map

# logging config
logging.basicConfig(
    format='%(message)s',
    filename='logs.log',
    level=logging.INFO,
)

@dataclass
class Driver:
    name: int
    num_actions: int
    graph: nx.DiGraph 
    is_logging: bool = False


    def __post_init__(self):
        self.rewards = []
        self.actions = []
        self.observations = []
        self.messages = []
        self.state_to_message = {
            0: "IDLE",
            1: "MATCHING", 
            2: "MATCHED", 
            3: "RIDING", 
            4: "OFF"
        }
        self.map = Map(self.graph)

    def action(self, observation) -> int:
        self.observations.append(observation)
        self._log_observation(observation)
        action = self._select_action(observation)
        self._log_action(action)
        self.actions.append(action)
        return action

    def _select_action(self, observation):
        state = self.state_to_message[observation[0]]
        position = observation[1]
        p_destination = observation[2]
        p_position = observation[3]
        price = observation[4]

        if state == 'IDLE':
            return position
        elif state == 'MATCHING':
            return 1
        elif state == 'MATCHED':
            if position == p_position:
                return position
            else:
                path = self.map.shortest_path(position, p_position)
                nxt = path[1]
                return nxt
        elif state == 'RIDING':
            if position == p_destination:
                return position
            else:
                path = self.map.shortest_path(position, p_destination)
                nxt = path[1]
                return nxt
        else:
            return position


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
        self.messages.append(f'Action = {action}')

    def _log_reward(self, reward):
        self.messages.append(f'Reward = {reward}')

    def _log_observation(self, observation):
        driver_msg = f'Driver {self.name}'
        state = self.state_to_message[observation[0]]
        state_msg = f'State = {state}'
        position_msg = f'Position = {observation[1]}'
        p_destination_msg = f'Passenger Destination = {observation[2]}'
        p_position_msg = f'Passenger Position = {observation[3]}'
        price_message = f'Price = {observation[4]}'

        message = driver_msg + '\t' + state_msg + '\t' + position_msg
        
        if state == "IDLE" or state == 'OFF':
            self.messages.append(message)
        elif state == "MATCHING":
            message += '\t' + p_destination_msg + '\t' + p_position_msg + '\t' + price_message
            self.messages.append(message)
        elif state == "MATCHED":
            message += '\t' + p_destination_msg + '\t' + p_position_msg
            self.messages.append(message)
        elif state == "RIDING":
            message += '\t' + p_destination_msg
            self.messages.append(message)

        return 