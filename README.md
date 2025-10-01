# ToF-Camera-Calibration

This repository contains the implementation code for the IROS 2025 paper **"An Easy Way for Extrinsic Calibration of Camera and Time-of-Flight Sensor"**.

## Overview

This project provides a simple and effective method for extrinsic calibration between a monocular camera and a Time-of-Flight (ToF) sensor. The calibration approach uses a specially designed target with three perpendicular planes to achieve accurate spatial alignment between the two sensors.

## Hardware Requirements

- **Monocular camera**: Intel RealSense camera (used in this implementation)
- **Time-of-Flight (ToF) sensor**: Any compatible ToF sensor
- **Fixture device**: Rigid mounting system to ensure camera and ToF sensor remain stationary during calibration
- **Calibration target**: 
  - 3 perpendicular planes configuration
  - Two vertical planes with white surfaces
  - Bottom plane with chessboard pattern starting exactly at the corner intersection
  - Detailed specifications can be found in the paper

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Tianyou-Nottingham/ToF.git
cd ToF
```

2. Install the required dependencies:


**Note**: Make sure to install all necessary packages for camera interface, ToF sensor communication, and data processing. Specific dependencies will vary based on your hardware configuration.

## Quick Start

### Step 1: Configuration
Modify the configuration settings in `./configs/config.py` to match your setup:
- ToF sensor serial number and parameters
- RealSense camera settings
- Save preferences and output directories
- Enable/disable specific code functions

### Step 2: Calibration Process
Run the main calibration script:
```bash
python calib.py
```

## File Structure and Components

### Core Calibration Files
- **`calib.py`** - Main calibration functions and workflow
- **`TOF_RANSAC.py`** - RANSAC algorithm implementation for ToF plane fitting
- **`two_plane_fit.py`** - Two-plane fitting algorithm for ToF sensor data
- **`read_data_utils.py`** - Initial processing utilities for ToF data
- **`obstacle_avoidance.py`** - ToF sensor obstacle detection functions

### Utility Files
- **`capture_RGB_ToF.py`** - Data collection for RGB+ToF applications (compatible with [DELTAR](https://github.com/zju3dv/deltar))
- **`plot.py`** - Visualization and plotting utilities
- **`./configs/config.py`** - Configuration file for all system parameters
- **`./utils/`** - Additional utility functions for various processing tasks

## Usage for RGB+ToF Applications

If you want to use the calibrated sensors for RGB+ToF depth estimation work (such as [DELTAR](https://github.com/zju3dv/deltar)), you can use the data collection approach provided in `capture_RGB_ToF.py`.

## Contributing

We welcome contributions to improve this calibration method. Please feel free to submit issues and pull requests.

## Citation

If you find this code useful for your research, please cite our paper:

```bibtex
@inproceedings{zhang2025easy,
  title={An Easy Way for Extrinsic Calibration of Camera and Time-of-Flight Sensor},
  author={Zhang, Tianyou and Liu, Jing and Dong, Xin and Axinte, Dragos},
  booktitle={IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
