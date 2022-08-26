from dataclasses import dataclass
import enum
import numpy as np
import gurobipy as grb
from typing import List

from ubergym.envs.maps import Map
from ubergym.envs.match_request import MatchRequest
from ubergym.envs.actors import Driver, Passenger
@dataclass
class Matcher:
    class Optimization(enum.Enum):
        LINEAR_SUM = 0
        LEXICOGRAPHIC_MINMAX = 1
        DYNAMIC = 2
    
    method: str
    MEAN_PRICE_PER_DISTANCE: float
    VARIANCE_PER_PRICE: float

    def __post_init__(self):
        if self.method == 'LINEAR_SUM':
            self.optimization = Matcher.Optimization.LINEAR_SUM
        elif self.method == 'LEXICOGRAPHIC_MINMAX':
            self.optimization = Matcher.Optimization.LEXICOGRAPHIC_MINMAX
        elif self.method == 'DYNAMIC':
            self.optimization = Matcher.Optimization.DYNAMIC
        else:
            raise ValueError(f'Unknown optimization method {self.method}')

    def match(self, 
        drivers: List[Driver],
        passengers: List[Passenger],
        map: Map) -> List[MatchRequest]:

        # if the matcher type is LINEAR_SUM or LEXICOGRAPHIC_MINMAX costs are the distances between waiting passengers and idle drivers
        # if the matcher type is DYNAMIC, costs are the distances between waiting passengers and idle + riding drivers
        # distance between a riding driver and a waiting passenger = distance to driver's dropoff + distance from the dropoff to the passenger 

        match_requests: List[MatchRequest] = []

        waiting_passengers = [i for i in range(len(passengers)) if passengers[i].status == Passenger.Status.WAITING]
        idle_drivers = [i for i in range(len(drivers)) if drivers[i].status == Driver.Status.IDLE]
        riding_drivers = [i for i in range(len(drivers)) if drivers[i].status == Driver.Status.RIDING]

        if self.optimization in [Matcher.Optimization.LINEAR_SUM, Matcher.Optimization.LEXICOGRAPHIC_MINMAX]:
            costs = np.zeros((len(waiting_passengers), len(idle_drivers)))  
            n, m = costs.shape

            if n == 0 or m == 0:
                return match_requests
            
            for i in range(n):
                for j in range(m):
                    # the function map.distance is lru cached, hence there are no recalculations
                    distance = map.distance(drivers[idle_drivers[j]].position, passengers[waiting_passengers[i]].position)
                    costs[i][j] = distance

        elif self.optimization == Matcher.Optimization.DYNAMIC:
            costs = np.zeros((len(waiting_passengers), len(idle_drivers) + len(riding_drivers)))  
            n, m = costs.shape

            if n == 0 or m == 0:
                return match_requests
            
            for i in range(n):
                for j in range(len(idle_drivers)):
                    # the function map.distance is lru cached, hence there are no recalculations
                    distance = map.distance(drivers[idle_drivers[j]].position, passengers[waiting_passengers[i]].position)
                    costs[i][j] = distance
            
            for i in range(n):
                for j in range(len(idle_drivers), m):
                    # the function map.distance is lru cached, hence there are no recalculations
                    d = drivers[riding_drivers[j - len(idle_drivers)]]
                    driver_position = d.position
                    driver_destination = passengers[d.passenger].destination
                    passenger_position = passengers[waiting_passengers[i]].position
                    distance = map.distance(driver_destination, passenger_position) + map.distance(driver_position, driver_destination)
                    costs[i][j] = distance

        solution = self.minimize_costs(costs)

        if solution is None:
            return match_requests

        for i in range(n):
            for j in range(m):

                if j >= len(idle_drivers):
                    driver = riding_drivers[j - len(idle_drivers)]
                else:
                    driver = idle_drivers[j]

                if solution[i,j] > 0.5:
                    passenger = passengers[waiting_passengers[i]]
                    price = self._price(distance = map.distance(passenger.position, passenger.destination))
                    match_requests.append(MatchRequest(driver = driver, passenger = waiting_passengers[i], price = price))
                    break
        
        return match_requests

    def _price(self, distance: int) -> float:
        mean_price = self.MEAN_PRICE_PER_DISTANCE * distance
        variance_price = self.VARIANCE_PER_PRICE * mean_price  
        price = max(0.0, np.random.normal(mean_price, variance_price))
        return price

    def minimize_costs(self, costs: np.ndarray) -> np.ndarray:
        """
        This function does optimal matching based on costs matrix and returns the matching solution.
        """

        # TODO: add something that lets go and doesn't match if there are way too many drivers and passengers so that optimization takes way too long

        #print(f'Optimization Method = {self.optimization.name}')

        if self.optimization == Matcher.Optimization.LINEAR_SUM:
            return self._linear_cost_minimization(costs)
        elif self.optimization == Matcher.Optimization.LEXICOGRAPHIC_MINMAX:
            return self._lexicographic_cost_minimization(costs)
        elif self.optimization == Matcher.Optimization.DYNAMIC:
            return self._dynamic_match(costs)
        else:
            return None

    def _linear_cost_minimization(self, costs: np.ndarray) -> np.ndarray:

        n,m = costs.shape
        
        model = grb.Model("matching")
        model.Params.LogToConsole = 0
        #model.Params.LogFile = "min_cost_matching.log"

        x = model.addMVar((n,m), vtype = grb.GRB.CONTINUOUS, name = "x")

        obj = grb.quicksum(costs[i][j] * x[i,j] for i in range(n) for j in range(m))
        model.setObjective(obj, grb.GRB.MINIMIZE)

        constr1 = grb.quicksum(x[i,j] for i in range(n) for j in range(m)) >= min(n,m)
        model.addConstr(constr1, name="Constr1")

        for i in range(n):
            model.addConstr(grb.quicksum(x[i,j] for j in range(m)) <= 1, name=f'Constr2_{i}')

        for j in range(m):
            model.addConstr(grb.quicksum(x[i,j] for i in range(n)) <= 1, name=f'Constr3_{j}')

        model.optimize()

        if model.status == grb.GRB.OPTIMAL:
            solution = np.array(model.getAttr('x')).reshape(n,m)
            return solution
        else:
            return None

    def _lexicographic_cost_minimization(self, costs: np.ndarray) -> np.ndarray:

        #print(f'Optimization Method = {self.optimization.name}')
        n,m = costs.shape

        # edit the costs with temperature for lexicographic min max 
        # set the temperature so that the maximum value of the new costs is MAXVAL

        # safe number for max value so that the algorithm still works
        MAXVAL = np.iinfo(np.int64).max / 1000.

        max_cost = np.max(costs)
        delta = np.log(MAXVAL) / max_cost
        # temperature = 1/delta
        #print('Temperature = {:.2e}'.format(temperature))

        for i in range(n):
            for j in range(m):
                costs[i][j] = np.exp(delta * costs[i][j])

        return self._linear_cost_minimization(costs)
    
    def _dynamic_match(self, costs: np.ndarray) -> np.ndarray:

        # this is different from the linear cost minimizer because the costs matrix includes riding drivers as well
        return self._linear_cost_minimization(costs)