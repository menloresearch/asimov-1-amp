# Copyright (c) 2026, Menlo Research.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Asimov-1 velocity tracking with an adversarial motion-prior reward."""

import gymnasium as gym

from . import agents

gym.register(
    id="Asimov1-Velocity-AMP",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.amp_env_cfg:Asimov1AmpEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Asimov1AMPRunnerCfg",
        "default_agent": "rsl_rl",
    },
)
