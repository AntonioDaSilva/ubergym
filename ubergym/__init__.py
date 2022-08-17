from gym.envs.registration import register

register(
    id='ubergym/uber-v0',
    entry_point='ubergym.envs:Uber'
)