# Neuromeka-IsaacLab

This repository is a Neuromeka extension for [IsaacLab](https://github.com/isaac-sim/IsaacLab). It provides robot assets, Gym environment registrations, task configurations, and RSL-RL scripts for Neuromeka simulation and reinforcement-learning workflows.

This repository is not limited to NAMI navigation. Current examples include:

- Manipulation tasks such as `Indy-Reach` and `Dual-Arm-Reach`
- Demo/deploy-style environments for Indy, Moby, NAMI, and Zen
- Shared environment, MDP, terrain, and utility modules for IsaacLab projects

## Compatibility

- Tested with Isaac Sim **5.1**
- Tested with IsaacLab **v2.3.0**
- Tested with the IsaacLab binary-install workflow using a prebuilt Isaac Sim installation

## Installation

### Prerequisites

Install Isaac Sim and IsaacLab first. Follow the IsaacLab v2.3.0 binary installation guide:

- [IsaacLab installation using Isaac Sim pre-built binaries](https://isaac-sim.github.io/IsaacLab/v2.3.0/source/setup/installation/binaries_installation.html)

After installing IsaacLab, activate the IsaacLab conda environment, for example:

```bash
conda activate env_isaaclab
```

Install [Git LFS](https://git-lfs.github.com/) before pulling the repository assets.

Install additional Python packages used by the examples:

```bash
pip install eclipse-zenoh open3d matplotlib pynput
```

### Installing The Extension

Clone this repository, pull LFS assets, and install it as an editable Python package in the IsaacLab environment:

```bash
conda activate env_isaaclab
cd nrmk_isaaclab_public
git lfs pull
pip install -e .
```

## Usage Examples

We recommend running these commands from the repository root with the IsaacLab conda environment activated.

### Training With RSL-RL

Train an Indy reach policy:

```bash
python scripts/rsl_rl/train.py --task Indy-Reach --num_envs 128 --headless --logger tensorboard
```

Train a dual-arm reach policy:

```bash
python scripts/rsl_rl/train.py --task Dual-Arm-Reach --num_envs 128 --headless --logger tensorboard
```

View training logs:

```bash
tensorboard --logdir logs/rsl_rl
```

### Playing A Trained Policy

Play a trained Indy reach checkpoint:

```bash
python scripts/rsl_rl/play.py --task Indy-Reach --num_envs 1 --checkpoint logs/rsl_rl/<experiment>/<run>/model_*.pt
```

If you want to use the checkpoint configured by the task runner, omit `--checkpoint`.

```bash
python scripts/rsl_rl/play.py --task Indy-Reach --num_envs 1
```

### Registered Environments

Trainable reach examples:

- `Indy-Reach`
- `Dual-Arm-Reach`

Demo/deploy-style task IDs for visualization, sensor streaming, and integration experiments:

- `Indy-Deploy`
- `Moby-Deploy`
- `Nami-Nav-Deploy`
- `Zen-Deploy`

The NAMI navigation demo config is in `isaac_neuromeka/tasks/demo/nami_env_cfg.py`. Demo video: <https://youtu.be/EHRZnBG3YPo>

## Custom Use Cases

Use these files as starting points when adding or modifying tasks:

- `isaac_neuromeka/tasks/manipulation/reach/indy/env_cfg.py`: Indy reach environment
- `isaac_neuromeka/tasks/manipulation/reach/dual_arm/env_cfg.py`: Dual-arm reach environment
- `isaac_neuromeka/tasks/manipulation/common/env_cfg_common.py`: Shared manipulation configuration
- `isaac_neuromeka/tasks/manipulation/reach/reach_env_cfg.py`: Shared reach-task configuration
- `isaac_neuromeka/tasks/demo`: Demo/deploy-style robot environments
- `scripts/rsl_rl/train.py`: RSL-RL training entry point
- `scripts/rsl_rl/play.py`: RSL-RL play/export entry point

For NAMI navigation scenes, update `NamiSceneCfg` in `isaac_neuromeka/tasks/demo/nami_env_cfg.py`. The provided scenes include:

- `isaac_neuromeka/assets/scene/hm3d_1`: [HM3D dataset](https://github.com/matterport/habitat-matterport-3dresearch)
- `isaac_neuromeka/assets/scene/hm3d_2`: [HM3D dataset](https://github.com/matterport/habitat-matterport-3dresearch)
- `isaac_neuromeka/assets/scene/nrmk_2nd_floor`: Neuromeka 2nd floor, scanned with [BLK2GO](https://shop.leica-geosystems.com/leica-blk/blk2go/overview)

## Setting Up VS Code

The repository includes an optional VS Code task that configures Python analysis paths for Isaac Sim and IsaacLab.

1. Open VS Code and press `Ctrl+Shift+P`.
2. Select **Tasks: Run Task**.
3. Run `setup_python_env`.
4. Enter the absolute paths to your Isaac Sim and IsaacLab installations when prompted.

By default, the task expects:

- Isaac Sim: `${HOME}/isaacsim`
- IsaacLab: `${HOME}/git/IsaacLab`

The task generates `.vscode/settings.json` so VS Code can index Isaac Sim, Omniverse, IsaacLab, and this extension.

## Credits

This project includes modified code from [IsaacLabExtensionTemplate](https://github.com/isaac-sim/IsaacLabExtensionTemplate), which is licensed under the MIT License. See `LICENSE-MIT` for details.

## Authors

- **Joonho Lee**, **Yunho Kim** at Neuromeka

## Acknowledgements

We sincerely appreciate the contributions of [Mayank Mittal](https://mayankm96.github.io/) for his support and insights in this project.
