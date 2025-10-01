# ToF-Camera-Calibration
Code of IROS 2025 paper "An Easy Way for Extrinsic Calibration of Camera and Time-of-Flight Sensor".

## Requirements
- One monocular camera (in this project is a RealSense).
- One Time-of-Flight (ToF) sensor.
- Fixture device to make sure camera and ToF sensor won't slip.
- Calibration target: 3 perpendicular planes, two vertical ones should be white and the bottom plane has chessboard starting from exactly the corner of three planes. Details can be found in the paper  
- Software requirments follow the instruction of code.


## Calibration
### Step 1
Change configuration in ```./config/config.py```, including ToF sensor serial and set-up, RealSense, save perference and code functions.

### Step 2
```calib.py``` includes main calibration functions.

```TOF_RANSAC.py``` includes RANSAC algorithm for ToF plane fitting.

```two_plane_fit.py``` includes ToF 2 planes fitting algorithm.

```read_data_utils.py``` includes functions initial processing of ToF data. 

```obstacle_avoidance.py``` includes functions for ToF sensor obstacel detection.

If you want to use the calibrated sensors for RGB+ToF depth estionmation work like [DELTAR](https://github.com/zju3dv/deltar), you can find the data collection approach in ```capture_RGB_ToF.py```

## Citation

If you find this code useful for your research, please use the following BibTeX entry.

```bibtex
@article{ToF_Camera_Calib,
  title={An Easy Way for Extrinsic Calibration of Camera and Time-of-Flight Sensor},
  author={Tianyou Zhang and Jing Liu and Xin Dong and Dragos Axinte},
  booktitle={IEEE/RSJ International Conferenceon Intelligent Robots and Systems (IROS)},
  year={2025}
}
```