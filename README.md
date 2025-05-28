# Neuromeka-IsaacLab

This is a project template that builds upon [*IsaacLab*](https://github.com/isaac-sim/IsaacLab). Before proceeding, please refer to the installation guides for IsaacLab.

We plan to add new environments and algorithms in future updates.

## Compatibility
- Tested with Isaac Sim **4.5.0** and IsaacLab **v2.0.1**

## Installation

### **Prerequisite: Install Git LFS**
Ensure that [Git LFS](https://git-lfs.github.com/) is installed before proceeding.

For detailed installation instructions, follow the [IsaacLab Installation Guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/binaries_installation.html). Additionally, for installing Isaac Sim, refer to [Isaac Sim Workstation Installation](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/install_workstation.html).

### **Installing the Extension**
Install this repository as a Python package. It will be installed with a symbolic link, ensuring that any modifications in the repository are reflected during execution.
```bash
cd neuromeka-isaac
pip install -e .
```

## Usage Examples

We recommend using a **Conda environment** from IsaacLab.

### **Training a Model**
Activate the Conda environment and start training:
```bash
conda activate env_isaaclab
python scripts/rsl_rl/train.py --task Indy-Reach --num_envs 4000 --headless --logger tensorboard
```

### **Playing a Trained Model**
```bash
conda activate env_isaaclab
python scripts/rsl_rl/play.py --task Indy-Reach --num_envs 1 
```

You can explore additional tasks in the `neuromeka-isaac/isaac_neuromeka/tasks` directory. For example:
- `neuromeka-isaac/isaac_neuromeka/tasks/manipulation/reach/dual_arm/__init__.py`
- `neuromeka-isaac/isaac_neuromeka/tasks/manipulation/reach/indy/__init__.py`

## Setting Up VSCode (Optional)

To configure VSCode for development, follow these steps:

1. Open VSCode and press `Ctrl+Shift+P`.
2. Select **Tasks: Run Task** and execute `setup_python_env` from the dropdown menu.
3. During execution, you will be prompted to enter the absolute path to your Isaac Sim installation.

This will generate a `.python.env` file in the `.vscode` directory, containing paths to all the Python modules provided by Isaac Sim and Omniverse. This enhances indexing and provides intelligent suggestions while coding.

*Note: Ensure your IsaacSim and IsaacLab paths are correctly configured in `tasks.json` or the prompt window.*

## Credits
This project includes modified code from [IsaacLabExtensionTemplate](https://github.com/isaac-sim/IsaacLabExtensionTemplate), which is licensed under the MIT License. See `LICENSE-MIT` for more details.

## Authors
- **Joonho Lee**, **Yunho Kim** at Neuromeka

## Acknowledgements
We sincerely appreciate the contributions of [Mayank Mittal](https://mayankm96.github.io/) for his support and insights in this project.
