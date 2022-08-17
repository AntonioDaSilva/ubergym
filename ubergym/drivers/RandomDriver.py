from typing import Iterable, List, Tuple
from dataclasses import dataclass
import logging, sys
import numpy as np

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

    def action(self, observation) -> int:
        self.observations.append(observation)
        self._log_observation(observation)
        action = np.random.randint(0, self.num_actions)
        self._log_action(action)
        self.actions.append(action)
        return action

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