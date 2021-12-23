from gym.envs.registration import register

register(
    id='ic_env-v0',
    entry_point='diffusion_gym.envs:ICEnv',
)