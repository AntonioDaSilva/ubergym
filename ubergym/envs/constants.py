"""
This file contains the parameters of the simulation.
"""

simulation = {
    "step_count": 0,
    "num_steps": 200,
    "mean_price_per_distance": 4.0,
    "variance_per_price": 0.2,
    "rewards": {
        "wait": 0.0,
        "move": -1.0,
        "accept_match": 0.0,
        "reject_match": 0.0,
        "arrive": 1.0,
        "invalid_action": 0,
    },
    "metadata": {"render_modes": [], "render_fps": None},
    "matcher_metadata": {"types": ['LINEAR_SUM', 'LEXICOGRAPHIC_MINMAX', 'DYNAMIC'], "default": 'LINEAR_SUM'},
    "is_logging": True,

}