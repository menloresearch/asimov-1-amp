# Asimov-1 AMP

Velocity tracking for the [Asimov-1](https://github.com/menloresearch/asimov-1) humanoid in
[Isaac Lab](https://github.com/isaac-sim/IsaacLab), with an adversarial motion prior (AMP) reward learned
from a reference walking clip.

The plain velocity task for Asimov-1 (`IsaacContrib-Velocity-Rough-Asimov1`) is being added to Isaac Lab in
[isaac-sim/IsaacLab#7071](https://github.com/isaac-sim/IsaacLab/pull/7071). This repository holds the AMP
extension: the discriminator, its PPO training loop, and the motion data.

## Status

- Isaac Lab 3.0 with the Isaac Sim PhysX backend. Newton and OvPhysX are not supported yet.
- Until isaac-sim/IsaacLab#7071 is released, this repository carries its own copy of the Asimov-1 robot
  configuration and base velocity task (`assets/asimov_1.py`, `tasks/velocity_amp/rough_env_cfg.py`,
  `tasks/velocity_amp/mdp/`). The copies match the pull request and will be replaced by imports from
  Isaac Lab once it ships.
- The robot asset is not hosted on the Isaac Lab asset server yet. Until it is, point the configuration at
  the URDF in [menloresearch/asimov-1](https://github.com/menloresearch/asimov-1), as shown below.

## Setup

```bash
git clone https://github.com/menloresearch/asimov-1.git
git clone https://github.com/menloresearch/asimov-1-amp.git
cd asimov-1-amp
export OMNI_KIT_ACCEPT_EULA=YES
uv sync
uv run pytest -q
```

## Train

```bash
uv run isaaclab train --rl_library rsl_rl --task Asimov1-Velocity-AMP --num_envs 4096 --visualizer none \
    env.scene.robot.spawn.asset_path=$(realpath ../asimov-1/sim-model/urdf/asimov_1.urdf)
```

Logs and checkpoints are written to `logs/rsl_rl/asimov_velocity_amp/`. The AMP terms are logged as
`Train/mean_amp_reward`, `Train/mean_task_reward`, and the `amp*` losses.

## Play

```bash
uv run isaaclab play --rl_library rsl_rl --task Asimov1-Velocity-AMP --checkpoint /abs/path/to/model.pt \
    env.scene.robot.spawn.asset_path=$(realpath ../asimov-1/sim-model/urdf/asimov_1.urdf)
```

Playback uses the task's `play_mode()`: 32 environments, no pushes or randomization, and no time-outs.

## How it works

- **Discriminator input.** The `amp` observation group holds the 23 joint positions (relative to the default
  pose) and joint velocities. The discriminator scores consecutive pairs `(s, s')`.
- **Expert data.** `tasks/velocity_amp/motions/asimov_1_walking.npz` is a 49 s walking clip at 50 Hz
  (2472 frames). It stores joint names, and the loader maps them onto the robot's joint order.
- **Reward.** The AMP reward is `coef * max(0, 1 - 0.25 * (D(s, s') - 1)^2)` and is blended with the task
  reward as `lerp * r_task + (1 - lerp) * r_amp`, with `coef = 0.3` and `lerp = 0.7`. The AMP reward is
  switched off for standing commands (command norm below 0.1).
- **Training.** The least-squares discriminator, with a gradient penalty on expert samples, is updated
  together with PPO in one optimizer. Policy transitions are kept in a replay buffer of 100k pairs.
  Pairs that cross an episode reset get no AMP reward and are left out of the buffer.

All AMP settings live in `AMPPPOAlgorithmCfg` in `tasks/velocity_amp/agents/rsl_rl_ppo_cfg.py`.

## Layout

```text
src/asimov_1_amp/
├── assets/asimov_1.py                 Asimov-1 articulation with delayed PD actuators
└── tasks/velocity_amp/
    ├── __init__.py                    registers Asimov1-Velocity-AMP
    ├── rough_env_cfg.py, mdp/         base velocity task (mirrors isaac-sim/IsaacLab#7071)
    ├── amp_env_cfg.py                 adds the `amp` observation group
    ├── agents/                        AMP PPO, discriminator, replay buffer, runner configuration
    └── motions/                       expert clip and loader
tests/test_amp.py                      unit tests for the AMP components
```

## Acknowledgements

The PPO update loop is adapted from [RSL-RL](https://github.com/leggedrobotics/rsl_rl) (BSD-3-Clause). The
AMP reward and discriminator follow [Peng et al., 2021](https://arxiv.org/abs/2104.02180) and
[Escontrela et al., 2022](https://arxiv.org/abs/2203.15103).

## License

BSD-3-Clause. See [LICENSE](LICENSE).
