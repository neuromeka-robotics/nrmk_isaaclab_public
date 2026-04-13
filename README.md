# Neuromeka-IsaacLab (NAMI navigation)
This repository contains minimal examples for navigating NAMI in a simulated environment. Specifically, it demonstrates:
- Creating a simulated environment using a scanned scene mesh file
- Spawning the NAMI robot within the environment
- Streaming RGB and depth data from the robot’s RGB-D sensor
- Controlling the NAMI robot using a keyboard

## Installation
### Prerequisite
#### 1. IsaacSim, IsaacLab
Follow the [IsaacLab Installation Guide](https://isaac-sim.github.io/IsaacLab/v2.2.1/source/setup/installation/binaries_installation.html) for IsaacSim and IsaacLab installation. 

The repository was tested with Isaac Sim **4.5.0** and IsaacLab **v2.2.1**.

For IsaacSim installation, we recommend binary installation rather than pip installation.

#### 2. Extra
After installing IsaacLab, a dedicated conda environment will be created (e.g., `env_isaaclab`).
Install extra packages.
```
conda activate env_isaaclab
pip install eclipse-zenoh open3d matplotlib pynput
```
Additionally, install [Git LFS](https://git-lfs.github.com/).

### Installing the neuromeka isaaclab extension
Clone the repository and install it as a package in the dedicated conda environment.
```
conda activate env_isaaclab
cd nrmk_isaaclab_public
pip install -e .
```

## Usage Examples ([Demo video](https://youtu.be/EHRZnBG3YPo))
Run simulator
```
python deploy/run_nami_sim.py
```
Run keyboard controller and visualizer
```
python deploy/run_nami_keyboard.py
python deploy/run_nami_keyboard.py --debug_vis  # To visualize image and pointcloud data
```

## Custom usecase
- In `deploy/run_nami_keyboard.py`, replace keyboard command with neural network controller or some other fancy algorithms.
- In `isaac_neuromeka/tasks/demo/nami_env_cfg.py`, change navigation scene with other opensource mesh file or manually scanned results. Check `NamiSceneCfg` in the file. Currently, below three scenes are provided.
    - `isaac_neuromeka/assets/scene/hm3d_1`: [HM3D dataset](https://github.com/matterport/habitat-matterport-3dresearch)
    - `isaac_neuromeka/assets/scene/hm3d_2`: [HM3D dataset](https://github.com/matterport/habitat-matterport-3dresearch)
    - `isaac_neuromeka/assets/scene/nrmk_2nd_floor`: Neuromeka 2nd floor (-> scanned with [BLK2GO](https://shop.leica-geosystems.com/leica-blk/blk2go/overview?c1=GAW_SE_NW&source=USA_RC_BRND&kw=blk2go_exm&utm_source=google&utm_medium=cpc&utm_term=blk2go_exm&utm_campaign=USA__-__Reality_Capture__-__Branded&cr5=773061516399&cr7=c&gad_source=1&gad_campaignid=20547366468&gbraid=0AAAAADnuiFisxx-3b-ZPtg_ZwbxQbzfSl&gclid=Cj0KCQjwv-LOBhCdARIsAM5hdKePRWJ0ettjodu7fIahSKQtW6kiOjItG4iWOboSDgRcVMvLCbLRnZgaAgQuEALw_wcB))

Core files to look into are as follows:
- `isaac_neuromeka/tasks/demo/__init__.py`: Gym environment definition
- `isaac_neuromeka/tasks/demo/nami_env_cfg.py`: NAMI environment
- `deploy/run_nami_sim.py`: Running simulator
- `deploy/run_nami_keyboard.py`: Running keyboard controller

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
