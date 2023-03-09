from gym.envs.registration import register

register(
    id='RollAndRake-v0',
    entry_point='roll_and_rake.envs:RollAndRakeEnvV0',
)

register(
    id='RollAndRake-v1',
    entry_point='roll_and_rake.envs:RollAndRakeEnvV1',
)

register(
    id='RollAndRake-v2',
    entry_point='roll_and_rake.envs:RollAndRakeEnvV2',
)