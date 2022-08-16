
@dataclass
class Driver:
    name: str
    learn: bool
    epsilon: float
    lr: float

    def __post_init__(self) -> None:
        self.log = False

    def compensate(self, s: simulation.Simulation) -> Iterable[Command]:
        cmds: List[Command] = []

        self._save_reward(s)
        self._log_reward(s)
        if self.learn:
            self._maybe_learn(s)

        for m in (
            self._maybe_create, self._maybe_reinforce
        ):
            cmds.extend(m(s))
        return cmds

    def initalize(self, s: simulation.Simulation, inital_position: int) -> None:
        self.inital_low = 0.0
        self.inital_high = 20.0 
        #self.lr = 0.05
        self.discount = 1.0
        self.n_price_state = 10
        self.max_price = 90
        self.qtable = dict()
        self.rewards = []
        #self.epsilon = 0.2

        '''
        we need a q table only in MATCHING AND IDLE STATES
        in MATCHED and RIDING states, the driver will move towards the simple goal

        in MATCHING state, the state = current position, passenger destination, passenger position, and price
        actions are 0 is reject and 1 is accept
        '''
        self.qtable[simulation.Driver.Status.IDLE] = np.random.uniform(self.inital_low, self.inital_high, (len(s.map), len(s.map)))
        self.qtable[simulation.Driver.Status.MATCHING] = np.random.uniform(self.inital_low, self.inital_high, (len(s.map), len(s.map), len(s.map), self.n_price_state, 2))
        self.qtable[simulation.Driver.Status.RIDING] = np.random.uniform(self.inital_low, self.inital_high, (len(s.map), len(s.map), len(s.map)))
        self.qtable[simulation.Driver.Status.MATCHED] = self.qtable[simulation.Driver.Status.RIDING]

        self.prev_state = DriverState(simulation.Driver.Status.IDLE, inital_position)
        self.prev_action = inital_position

        # initalize non-neighbor movements to -np.inf so that they are not chosen
        for i in range(len(s.map)):
            for j in range(len(s.map)):
                if i == j:
                    continue
                if s.map.neighbors(i,j):
                    continue
                self.qtable[simulation.Driver.Status.IDLE][i][j] = -np.inf

        for i in range(len(s.map)):
            for j in range(len(s.map)):
                for k in range(len(s.map)):
                    if i == k:
                        continue
                    if s.map.neighbors(i,k):
                        continue
                    self.qtable[simulation.Driver.Status.RIDING][i][j][k] = -np.inf

    def _save_reward(self, s: simulation.Simulation) -> None:
        if s.map is None:
            return 
        if self.name not in s.drivers:
            return 
        if self.name not in s.rewards:
            return
        # TODO: change all of these into constants
        if s.clock - simc.simulation_clock_step not in s.rewards[self.name]:
            return

        reward = s.rewards[self.name][s.clock - simc.simulation_clock_step]
        self.rewards.append(reward)

    def _log_reward(self, s: simulation.Simulation) -> None:
        if s.map is None:
            return 
        if self.name not in s.drivers:
            return 

        if self.log:
            logger.info(f'Driver {self.name}',rewards=self.rewards[-min(10, len(self.rewards)):])

        return 

    def _get_current_state(self, s: simulation.Simulation) -> DriverState:

        d = s.drivers[self.name]
        if d.status == simulation.Driver.Status.IDLE:
            return DriverState(d.status, (d.position,))

        p = s.passengers[d.match_request.passenger]
        price = d.match_request.price
        encoded_price = self._encode_price(price)

        if d.status == simulation.Driver.Status.MATCHING:
            return DriverState(d.status, (d.position, p.destination, p.position, encoded_price))

        if d.status == simulation.Driver.Status.MATCHED:
            return DriverState(d.status, (d.position, p.position))

        if d.status == simulation.Driver.Status.RIDING:
            return DriverState(d.status, (d.position, p.destination))

    def _select_random_action(self, state: DriverState) -> int:
        n_actions = self.qtable[state.status][state.index].shape[0]
        action = np.random.randint(0, n_actions)
        best_action = np.argmax(self.qtable[state.status][state.index])
        while self.qtable[state.status][state.index][action] == -np.inf:
            action = np.random.randint(0, n_actions)
        
        if self.log:
            logger.info(f'Driver {self.name}',n_actions=n_actions)
            logger.info(f'Driver {self.name}', random_action=action)
        return action

    def _maybe_learn(self, s: simulation.Simulation) -> None:
        if s.map is None:
            return 
        if self.name not in s.drivers:
            return 
        if self.name not in s.rewards:
            return
        # TODO: change all of these into constants
        if s.clock - simc.simulation_clock_step not in s.rewards[self.name]:
            return
        
        reward = s.rewards[self.name][s.clock - simc.simulation_clock_step]
        
        # TD learning
        current_state = self._get_current_state(s)
        qmax = np.max(self.qtable[current_state.status][current_state.index])
        qcurr = self.qtable[self.prev_state.status][self.prev_state.index][self.prev_action]
        self.qtable[self.prev_state.status][self.prev_state.index][self.prev_action] += self.lr * (reward + self.discount * qmax - qcurr)

        if self.log:
            logger.info(f'Driver {self.name}',reward=reward)
            logger.info(f'Driver {self.name}',prev_state=self.prev_state)
            logger.info(f'Driver {self.name}',current_state=current_state)
            logger.info(f'Driver {self.name}',qold=qcurr)
            logger.info(f'Driver {self.name}',qnew=self.qtable[self.prev_state.status][self.prev_state.index][self.prev_action])

    def _maybe_create(self, s: simulation.Simulation) -> Iterable[Command]:
        if s.map is None:
            return ()
        if self.name not in s.drivers:
            inital_position = s.map.random_node()
            self.initalize(s, inital_position)
            return (commands.CreateDriver(name=self.name, position=inital_position),)
        return ()

    def _maybe_reinforce(self, s: simulation.Simulation) -> Iterable[Command]:
        '''
        the driver uses his q table to decide actions
        '''
        if s.map is None:
            return ()
        if self.name not in s.drivers:
            return ()

        #self.epsilon -= self.epsilon/(simc.n_steps + 1)

        p = np.random.random()
        if p < self.epsilon:
            # explore
            self.random_action = True
        else:
            self.random_action = False

        d = s.drivers[self.name]
        state = self._get_current_state(s)
        self.prev_state = state

        if self.log:
            logger.info(f'Driver {self.name}', state=repr(state))
            logger.info(f'Driver {self.name}', qtable=self.qtable[state.status][state.index])


        if state.status == simulation.Driver.Status.MATCHING:
            if self.random_action:
                action = self._select_random_action(state)
            else:
                action = np.argmax(self.qtable[state.status][state.index])
            self.prev_action = action

            if self.log:
                logger.info(f'Driver {self.name}', action=action)

            return (commands.RespondMatch(driver=self.name, accept=bool(action)),)

        if state.status in [simulation.Driver.Status.IDLE, simulation.Driver.Status.MATCHED, simulation.Driver.Status.RIDING]:
            curr = state.index[0]

            if self.random_action:
                dst = self._select_random_action(state)
            else:
                dst = np.argmax(self.qtable[state.status][state.index])
            
            self.prev_action = curr

            # the engine will reject commands which move too fast
            if d.can_move(s, dst):
                self.prev_action = dst
            
            if self.log:
                logger.info(f'Driver {self.name}', action=self.prev_action)

            if self.prev_action != curr:
                return (commands.MoveDriver(name=self.name, position=self.prev_action),)

            return ()


        return ()

    def _encode_price(self, price: float) -> int:
        if price >= self.max_price:
            return self.n_price_state - 1
        elif price < 0:
            return 0
        else:
            return int(price // int(self.max_price / self.n_price_state))

    def summarize_q_table(self):
        """
        Summarize the q table.
        """

        qtable = self.qtable
        for key in qtable.keys():
            print(f"State: {key}\n")
            if key == simulation.Driver.Status.IDLE:
                for i in range(len(qtable[key])):
                    action = np.argmax(qtable[key][i])
                    print(f"{i} -> {action}")

            elif key == simulation.Driver.Status.MATCHING:
                for i in range(len(qtable[key])):
                    for j in range(len(qtable[key][i])):
                        for k in range(len(qtable[key][i][j])):
                            if j == 0 and k == 1:
                                action = np.argmax(qtable[key][i][j][k][-1])
                                print(f"{i} {j} {k} {90} -> {action}")
            
            elif key == simulation.Driver.Status.RIDING or key == simulation.Driver.Status.MATCHED:
                for i in range(len(qtable[key])):
                    for j in range(len(qtable[key][i])):
                        action = np.argmax(qtable[key][i][j])
                        print(f"{i} {j} -> {action}")

            print("\n")

    def reset(self):
        self.rewards = []
        self.prev_state = None
        self.prev_action = None