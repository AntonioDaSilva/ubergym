import gym
from gym import spaces

import networkx as nx
import numpy as np
from typing import Optional, List, Tuple
from collections import OrderedDict
import logging, sys

from ubergym.envs.maps import Map
from ubergym.envs.actors import Driver, Passenger
from ubergym.envs.matcher import Matcher
from ubergym.envs.match_request import MatchRequest
import ubergym.envs.constants as constants

# logging config
logging.basicConfig(
    format='%(message)s',
    filename='logs.log',
    level=logging.INFO,
)

class Uber(gym.Env):
    metadata = constants.simulation["metadata"]
    matcher_metadata = constants.simulation["matcher_metadata"]
    
    def __init__(
        self, 
        n_drivers: int, 
        passenger_generation_probabilities: np.ndarray, 
        graph: nx.DiGraph, 
        num_steps: Optional[int] = constants.simulation["num_steps"], 
        matcher_type: Optional[str] = None, 
        is_logging: Optional[bool] = constants.simulation["is_logging"], 
        seed: Optional[int] = None, 
        render_mode: Optional[str] = None) -> None:
        
        if not(render_mode is None or render_mode in self.metadata["render_modes"]):
            raise ValueError(f"render_mode {render_mode} is not supported")

        self.render_mode = render_mode
        self.is_logging = is_logging
        self.messages = []

        self.SEED = seed
        np.random.seed(self.SEED)
        
        self.n_drivers = n_drivers

        self.edge_weight = self._check_graph(graph)
        
        self.map = Map(graph)

        if len(passenger_generation_probabilities) != len(self.map):
            raise ValueError("passenger_generation_probabilities must have length equal to the number of nodes in the graph")
        
        self.passenger_generation_probabilities = passenger_generation_probabilities

        self.observation_space = spaces.Dict(
            {
                "state": spaces.MultiDiscrete([len(Driver.Status)] * self.n_drivers),
                "position": spaces.MultiDiscrete([len(self.map)] * self.n_drivers),
                "passenger_destination": spaces.MultiDiscrete([len(self.map)] * self.n_drivers),
                "passenger_position": spaces.MultiDiscrete([len(self.map)] * self.n_drivers),
                "price": spaces.Box(low=0, high=np.inf, shape=(self.n_drivers,), dtype=np.float64),
            }
        )

        # all of the actions can be encoded as an integer between 0 and len(self.map) - 1 even though most of these actions are invalid
        self.action_space = spaces.MultiDiscrete([len(self.map)] * self.n_drivers)

        self.step_count = constants.simulation["step_count"]
        self.step_size = self.edge_weight
        self.num_steps = num_steps

        # initialize drivers and passengers
        self.drivers: List[Driver] = []
        for i in range(self.n_drivers):
            self.drivers.append(Driver(last_move = self.step_count, name = i, position = self.map.random_node(), status = Driver.Status.IDLE))

        self.passengers: List[Passenger] = []

        # initialize matcher
        if not(matcher_type is None or matcher_type in self.matcher_metadata["types"]):
            raise ValueError(f"matcher_type {matcher_type} is not supported")

        if matcher_type is None:
            matcher_type = self.matcher_metadata["default"]
        self.matcher_type = matcher_type
        self.MEAN_PRICE_PER_DISTANCE = constants.simulation["mean_price_per_distance"]
        self.VARIANCE_PER_PRICE = constants.simulation["variance_per_price"]
        self.matcher = Matcher(self.matcher_type, self.MEAN_PRICE_PER_DISTANCE, self.VARIANCE_PER_PRICE)

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool, dict]:

        self.step_count += self.step_size
        rewards = self._process_actions(actions)
        self._generate_passengers()
        self._send_match_requests()
        if self.is_logging:
            self._log()

        done = self.step_count == self.num_steps * self.step_size
        info = self._get_info()
        observation = self._get_obs()
        return observation, rewards, done, info 
    
    def reset(self, seed=None, return_info=False, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self.step_count = 0
        self.passengers = []
        self._generate_passengers()
        self.drivers = []
        for i in range(self.n_drivers):
            self.drivers.append(Driver(last_move = self.step_count, name = i, position = self.map.random_node(), status = Driver.Status.IDLE))

        observation = self._get_obs()
        info = self._get_info()
        return (observation, info) if return_info else observation

    def render(self):
        return ()

    def _process_actions(self, actions: np.ndarray) -> np.ndarray:
        """
        Process actions and update the state accordingly.
        """
        if len(actions) != self.n_drivers:
            raise ValueError("actions must have length equal to the number of drivers")

        rewards = np.zeros(self.n_drivers)

        for i in range(self.n_drivers):
            d = self.drivers[i]
            a = actions[i]
            if d.status in [Driver.Status.IDLE, Driver.Status.MATCHED, Driver.Status.RIDING]:
                rewards[i] += self._move_driver(i, a)    
            elif d.status == Driver.Status.MATCHING:
                rewards[i] += self._match_driver(i, a)

        return rewards

    def _match_driver(self, i: int, a: int) -> float:
        """
        Match a driver with a passenger if conditions are satisfied.
        """
        reward = 0.0

        if a == 0:
            d = self.drivers[i]
            p = self.passengers[d.match_request.passenger]
            d.match_request = None
            d.status = Driver.Status.IDLE
            p.status = Passenger.Status.WAITING
            p.driver = None
            reward += constants.simulation["rewards"]["reject_match"]
        
        elif a != 1:
            reward += constants.simulation["rewards"]["invalid_action"]
        
        else:
            # case of accept match
            d = self.drivers[i]
            p = self.passengers[d.match_request.passenger]
            d.status = Driver.Status.MATCHED
            p.status = Passenger.Status.MATCHED
            p.driver = i
            d.passenger = d.match_request.passenger
            reward += constants.simulation["rewards"]["accept_match"]

            # if the match happens in the same location, automatically start the ride
            if d.position == p.position:
                p.status = Passenger.Status.RIDING
                d.status = Driver.Status.RIDING

        self._log_match(i, a)
        return reward

    def _move_driver(self, driver: int, destination: int) -> float:
        """
        Move a driver to a new position if conditions are satisfied.
        """
        reward = 0.0

        d = self.drivers[driver]

        if destination == d.position:
            reward += constants.simulation["rewards"]["wait"]
            return reward
        if not self.map.neighbors(d.position, destination):
            reward += constants.simulation["rewards"]["invalid_action"]
            return reward 

        distance = self.map.graph[d.position][destination]['weight']
        if distance > self.step_count - d.last_move:
            reward += constants.simulation["rewards"]["wait"]
            return reward 

        arrival = False
        pickup = False
        prev_position = d.position
        new_position = destination
        passenger = None

        d.position = destination
        d.last_move = self.step_count
        reward += distance * constants.simulation["rewards"]["move"]
        
        # if the driver has a passenger, move the passenger as well
        if d.status == Driver.Status.RIDING:
            p = self.passengers[d.passenger]
            p.position = destination
            # case of arrival at passenger's destination
            if p.position == p.destination:
                arrival = True
                passenger = d.passenger
                reward += d.match_request.price * constants.simulation["rewards"]["arrive"]
                d.match_request = None
                d.status = Driver.Status.IDLE
                d.passenger = None
                p.status = Passenger.Status.ARRIVED
                p.driver = None
                p.arrived_at = self.step_count
        
        # if the driver is matched with a passenger, pick up the passenger upon arrival
        elif d.status == Driver.Status.MATCHED:
            p = self.passengers[d.passenger]
            if p.position == d.position:
                pickup = True
                passenger = d.passenger
                p.status = Passenger.Status.RIDING
                d.status = Driver.Status.RIDING
                p.picked_up_at = self.step_count

        self._log_move(driver, prev_position, new_position, arrival, pickup, passenger)

        return reward

    def _get_obs(self) -> OrderedDict:
        """
        Get an observation from the current state.
        """
        d = OrderedDict()
        d['state'] = np.array([d.status.value for d in self.drivers])
        d['position'] = np.array([d.position for d in self.drivers])
        d['passenger_destination'] = np.array([self.passengers[d.passenger].destination if d.passenger is not None else 0 for d in self.drivers])
        d['passenger_position'] = np.array([self.passengers[d.passenger].position if d.passenger is not None else 0 for d in self.drivers])
        d['price'] = np.array([d.match_request.price if d.match_request is not None else 0 for d in self.drivers])
        return d

    def _get_info(self) -> dict:
        """
        Get information about the current state.
        """
        info = {}
        info["step_count"] = self.step_count
        return info

    def _generate_passengers(self) -> None:

        np.random.seed(self.SEED)
        
        for i in range(len(self.map)):
            if np.random.random() < self.passenger_generation_probabilities[i]:
                destination = self.map.random_node()
                while destination == i:
                    destination = self.map.random_node()
                name = len(self.passengers)
                self.passengers.append(Passenger(name = name, position = i, destination = destination, status = Passenger.Status.WAITING, spawned_at = self.step_count))
                self._log_passenger_generation(name, i, destination)
        return

    def _send_match_requests(self) -> None:

        match_requests = self._generate_match_requests()
        for match_request in match_requests:
            d = self.drivers[match_request.driver]
            p = self.passengers[match_request.passenger]
            if d.status != Driver.Status.IDLE or p.status != Passenger.Status.WAITING:
                continue

            d.status = Driver.Status.MATCHING
            d.match_request = match_request
            p.status = Passenger.Status.MATCHING
            self._log_match_request(match_request)


    def _generate_match_requests(self) -> List[MatchRequest]:

        return self.matcher.match(self.drivers, self.passengers, self.map)

    def _check_graph(self, graph):

        # only allow nx.DiGraph
        if not type(graph) is nx.DiGraph:
            raise TypeError("graph should be of type nx.DiGraph")

        # only allow strongly connected graphs
        if not nx.is_strongly_connected(graph):
            raise ValueError("graph is not strongly connected")
        
        # only allow constant weights
        weight = None
        for u,v,d in graph.edges(data=True):
            try:
                edge_weight = d["weight"]
            except:
                raise ValueError("edges of the graphs should have the weight attribute")

            if weight is None:
                weight = edge_weight
            elif edge_weight != weight:
                raise ValueError("edges of the graph should have constant weight")

        return weight


    def _log(self):
        if not self.is_logging:
            return
        
        for message in self.messages:
            logging.info(message)
        
        self.messages = []

    def _log_match(self, driver: int, response: int):

        if not(response == 0 or response == 1):
            return

        if response == 0:
            message = f"Driver {driver} rejected match request."
        elif response == 1:
            message = f"Driver {driver} accepted match request."

        self.messages.append(message)

    def _log_move(self, driver: int, prev_position: int, new_position: int, arrival: bool = False, pickup: bool = False, passenger: int = None):
        
        self.messages.append(f"Driver {driver} moved from {prev_position} to {new_position}.")

        if arrival:
            self.messages.append(f"Driver {driver} arrived at passenger {passenger}'s destination.")
        
        if pickup:
            self.messages.append(f"Driver {driver} picked up the passenger {passenger}.")

    def _log_match_request(self, match_request: MatchRequest):

        self.messages.append(f"Match Request for driver {match_request.driver}, passenger {match_request.passenger} with price {match_request.price}.")

    def _log_passenger_generation(self, passenger: int, position: int, destination: int):

        self.messages.append(f"Passenger {passenger} generated at position {position} to destination {destination}.")