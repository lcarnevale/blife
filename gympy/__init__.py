from gym.envs.registration import register

register(
    id='blife-v0',
    entry_point='gympy.envs:BatteryLifetimeEnv',
)