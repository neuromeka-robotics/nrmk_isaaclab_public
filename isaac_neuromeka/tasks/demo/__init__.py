import gymnasium as gym

from . import indy_env_cfg, moby_env_cfg, learning
##
# Register Gym environments.
##

gym.register(
    id="Indy-Deploy",
    # entry_point="isaac_neuromeka.env.rl_task_custom_env:CustomManagerBasedRLEnv",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": indy_env_cfg.IndyDeployEnvCfg,
        "rsl_rl_cfg_entry_point": f"{learning.__name__}.rsl_rl_cfg:ReachPPORunnerCfg",
        "nrmk_rl_cfg_entry_point": f"{learning.__name__}.nrmk_rl_cfg:ReachPPORunnerCfg",
    },
)

gym.register(
    id="Indy-Deploy-Debug",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": indy_env_cfg.IndyDeployEnvRSLRL,
        "rsl_rl_cfg_entry_point": f"{learning.__name__}.rsl_rl_cfg:ReachPPORunnerCfg",
        "nrmk_rl_cfg_entry_point": f"{learning.__name__}.nrmk_rl_cfg:ReachPPORunnerCfg",
    },
)

gym.register(
    id="Moby-Deploy",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": moby_env_cfg.MobyDeployEnvCfg,
        "rsl_rl_cfg_entry_point": f"{learning.__name__}.rsl_rl_cfg:ReachPPORunnerCfg",
        "nrmk_rl_cfg_entry_point": f"{learning.__name__}.nrmk_rl_cfg:ReachPPORunnerCfg",
    },
)
