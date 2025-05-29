import gymnasium as gym

from . import indy_env_cfg, learning
##
# Register Gym environments.
##

gym.register(
    id="Indy-Stream",
    entry_point="isaac_neuromeka.env.rl_task_custom_env:CustomManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": indy_env_cfg.IndyStreamEnvCfg,
        "rsl_rl_cfg_entry_point": f"{learning.__name__}.rsl_rl_cfg:ReachPPORunnerCfg",
        "nrmk_rl_cfg_entry_point": f"{learning.__name__}.nrmk_rl_cfg:ReachPPORunnerCfg",
    },
)
