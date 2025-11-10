import gymnasium as gym
from . import agents

# Register Gym environments
gym.register(
    id="ArmFollow-Direct-v0",  # ← simpler, clearer name
    entry_point=f"{__name__}.arm_follow_env:ArmFollowEnv",
    disable_env_checker=True,
    kwargs={
        # path to your cfg class
        "env_cfg_entry_point": f"{__name__}.arm_follow_env_cfg:ArmFollowEnvCfg",

        # this tells IsaacLab which RL config to load (RSL-RL PPO by default)
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)

