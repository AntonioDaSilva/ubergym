from dataclasses import dataclass, field
import enum
from typing import Optional

@dataclass
class Passenger:
    class Status(enum.Enum):
        WAITING = 0
        MATCHING = 1
        MATCHED = 2
        RIDING = 3
        ARRIVED = 4

    name: int
    position: int
    destination: int
    status: Status
    spawned_at: int
    picked_up_at: Optional[int] = None
    arrived_at: Optional[int] = None
    driver: Optional[int] = None
    

from ubergym.envs.match_request import MatchRequest

@dataclass
class Driver:
    class Status(enum.Enum):
        IDLE = 0
        MATCHING = 1
        MATCHED = 2
        RIDING = 3
        OFF = 4

    last_move: int  # clock at time of last move
    name: int
    position: int
    status: Status
    passenger: Optional[int] = field(default=None)
    match_request: Optional[MatchRequest] = field(default=None)