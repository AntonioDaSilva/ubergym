from dataclasses import dataclass
import enum
import numpy as np
import gurobipy as grb

@dataclass
class Matcher:
    class Optimization(enum.Enum):
        LINEAR_SUM = 0
        LEXICOGRAPHIC_MINMAX = 1
    
    method: str = 'LINEAR_SUM'
    MEAN_PRICE_PER_DISTANCE: float = 5.0
    VARIANCE_PER_PRICE: float = 0

    def __post_init__(self):
        if self.method == 'LINEAR_SUM':
            self.optimization = Matcher.Optimization.LINEAR_SUM
        elif self.method == 'LEXICOGRAPHIC_MINMAX':
            self.optimization = Matcher.Optimization.LEXICOGRAPHIC_MINMAX
        else:
            raise ValueError(f'Unknown optimization method {self.method}')

    def minimize_costs(self, costs: np.ndarray) -> np.ndarray:
        """
        This function does optimal matching based on costs matrix and returns the matching solution.
        """

        # TODO: add something that lets go and doesn't match if there are way too many drivers and passengers so that optimization takes way too long

        #print(f'Optimization Method = {self.optimization.name}')
        n,m = costs.shape

        if self.optimization == Matcher.Optimization.LINEAR_SUM:
            pass

        elif self.optimization == Matcher.Optimization.LEXICOGRAPHIC_MINMAX:
            # edit the costs with temperature for lexicographic min max 
            # set the temperature so that the maximum value of the new costs is MAXVAL

            # safe number for max value so that the algorithm still works
            MAXVAL = np.iinfo(np.int64).max / 1000.

            max_cost = np.max(costs)
            delta = np.log(MAXVAL) / max_cost
            temperature = 1/delta
            #print('Temperature = {:.2e}'.format(temperature))

            for i in range(n):
                for j in range(m):
                    costs[i][j] = np.exp(delta * costs[i][j])


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

