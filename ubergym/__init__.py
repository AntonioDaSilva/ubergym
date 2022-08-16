from gym.envs.registration import register

register(
    id='uber_gym/uber-v0',
    entry_point='uber_gym.envs:Uber',
    max_episode_steps=300,
)