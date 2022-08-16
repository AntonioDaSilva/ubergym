from dataclasses import dataclass

@dataclass
class MatchRequest:
    driver: int
    passenger: int
    price: float