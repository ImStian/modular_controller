# Modular Controller

Modular ROS2 controller for ASV-towfish cable towing system with waypoint-based path following.

## Architecture

```
Waypoints → Path Generator → LOS Guidance → MRAC Controller → Heading Controller → Thrust Allocator
```

## Features

- **Waypoint Path Following**: Linear, cubic, or B-spline interpolation
- **LOS Guidance**: Line-of-sight path following with cross-track error control
- **MRAC Controller**: Model Reference Adaptive Controller for pendulum dynamics
- **Heading Controller**: PD controller with feedforward, three modes (PATH, LOS, FORCE)
- **Thrust Allocation**: Differential thrust for underactuated ASV

## Video Showcase
The following videos demonstrates the project. The tracking is not fenomenal.
- [Potato Shaped Path](https://youtu.be/pNIBTUkByVg?si=Ot8wUOWTktnUME-A)  
- [Sinusoidal Path](https://www.youtube.com/watch?v=AeA2Gf_tDqs)
- [Straight Path](https://youtu.be/S3Da_DntyDQ)

## Installation Prerequisites
The control system is made for use with [Marine Robotics Simulation Framework by Markus Buchholz](https://github.com/markusbuchholz/marine-robotics-sim-framework). It is assumed you have already installed and built this for the installation of the Towing Controller. The installation guide for this codebase will start at the "Run Docker"-step from Marine Robotics Simulation Framework as some alterations are required.

## Install and Running Docker Container
### Step 1:
Put repository contents inside a folder 'modular_controller' and place this inside Marine Robotics Simulation Framework environment:
```bash 
marine-robotics-sim-framework/gz_ws/src/modular_controller/
```
### Step 2:
For first time installation: Adjust in ```run.sh```.
```bash
local_gz_ws="/home/markus/underwater/marine-robotics-sim-framework/gz_ws"
local_SITL_Models="/home/markus/underwater/marine-robotics-sim-framework/SITL_Models"
```
You can now run the Docker Container with the Modular Controller with a similar step to the one outlined in [Marine Robotics Framework by Markus Buchholz](https://github.com/markusbuchholz/marine-robotics-sim-framework).
```bash
sudo ./run.sh
```
Build and source the ROS2 environments:
```bash
colcon build
source install/setup.bash

cd ../gz_ws
colcon build --symlink-install --merge-install --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo -DBUILD_TESTING=ON -DCMAKE_CXX_STANDARD=17

source install/setup.bash
source gazebo_exports.sh

pip install autograd
pip install --upgrade matplotlib
pip install jax
```

## Adding Required Odometry Publisher:
The controller assumes that the position of the towfish (ROV) is known. By default the odometry of the ROV is not published by the simulator and must to be manually added to the simulator's ROV model file.
This is done by adding the segment below to the `model.sdf` file in `/marine-robotics-sim-framework/SITL_Models/Gazebo/models/bluerov2_heavy/` before the thruster definitions (around line 117).
```html
    <!-- plugins -->
    <plugin name="gz::sim::systems::JointStatePublisher"
      filename="gz-sim-joint-state-publisher-system">
    </plugin>
    <plugin name="gz::sim::systems::OdometryPublisher"
      filename="gz-sim-odometry-publisher-system">
      <odom_frame>odom</odom_frame>
      <robot_base_frame>base_link</robot_base_frame>
      <dimensions>3</dimensions>
    </plugin>
```


## Usage
Running the controller requires the simulator running in the background.

### Open Multiple Docker Containers
You can always connect to running Docker containers from other terminals:
```bash
sudo docker exec -it marine_robotics_sitl /bin/bash
```

### Launch Simulator:
```bash
ros2 launch move_blueboat multirobot_mission_simulation.launch.py
```
**Note:** For running simulations outside of the default boundingbox of the simulator (-17.25 x 17.25 meters), manually remove the 'wind_turbine' and 'bop_panel' from the simulation environment within Gazebo.

### Launching Towing Controller:
The following command will launch the ASV-Towfish Controller with configurable parameters.
```bash
ros2 launch modular_controller modular_controller_with_plotter.launch.py waypoints_file:=straight_line.yaml los_speed:=1.25 los_delta:=2.0 los_k:=1.5 k_psi:=40.0 k_r:=15.0
```
**Note:** Parameter numbers <u>must</u> be floats!
### Parameters:

Edit `config/waypoints.yaml` to define your mission:

```yaml
waypoints:
  - [0.0, 0.0]
  - [10.0, 0.0]
  - [10.0, 10.0]
  - [0.0, 10.0]

path_settings:
  interpolation: 'cubic'  # 'linear', 'cubic', or 'bspline'
  resolution: 100
  closed_path: true
```
**Predefined paths:**
- `circle.yaml`, `straight_line.yaml`, `u_shape.yaml`, `wave_closed_loop.yaml`, `wave_high_freq.yaml`, `wave_large_amplitude.yaml`, `waypoints.yaml`
 
**Heading Modes:**
This parameter must be adjusted in `controller_node.py`:
- `FORCE` - ASV points in the direction of the MRAC Force Command
- `LOS` - ASV points in the direction of the velocity vector from LOS
- `PATH` - ASV points in the tanget to the path at the current position.

**ROS2 parameters:**
- `waypoints_file`: Path to waypoints YAML file
- `los_speed`: Desired speed [m/s] 
- `los_delta`: Look-ahead distance [m] 
- `los_k`: Convergence gain (default: 0.5)
- `heading_mode`: 'path', 'los', or 'force' 
- `k_psi`: Heading error gain [N⋅m/rad]
- `k_r`: Yaw rate damping [N⋅m⋅s/rad] 



## Licence:
Please source me if you decide to build ontop of this project. :)
