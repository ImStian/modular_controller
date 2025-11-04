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
- Potato Shaped Path : https://youtu.be/pNIBTUkByVg?si=Ot8wUOWTktnUME-A
- Sinusoidal Path : https://www.youtube.com/watch?v=AeA2Gf_tDqs

## Installation

### Build with colcon

```bash
cd ~/your_workspace
colcon build --packages-select modular_controller
source install/setup.bash
```

## Usage

### With Simulator

Launch controller with ASV-towfish simulator:

```bash
ros2 launch modular_controller modular_controller_with_sim.launch.py
```

### Standalone

Launch controller only (expects odometry from Gazebo or hardware):

```bash
ros2 launch modular_controller modular_controller.launch.py waypoints_file:=waypoints.yaml
```

### Parameters

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

ROS2 parameters:
- `waypoints_file`: Path to waypoints YAML file
- `los_speed`: Desired speed [m/s] (default: 1.0)
- `los_delta`: Look-ahead distance [m] (default: 5.0)
- `los_k`: Convergence gain (default: 0.5)
- `heading_mode`: 'path', 'los', or 'force' (default: 'los')
- `k_psi`: Heading error gain [N⋅m/rad] (default: 10.0)
- `k_r`: Yaw rate damping [N⋅m⋅s/rad] (default: 5.0)

## Topics

### Subscriptions
- `/model/blueboat/odometry` (nav_msgs/Odometry): ASV state
- `/model/bluerov2_heavy/odometry` (nav_msgs/Odometry): Towfish state

### Publications
- `/model/blueboat/joint/motor_port_joint/cmd_thrust` (std_msgs/Float32): Left thruster command
- `/model/blueboat/joint/motor_stbd_joint/cmd_thrust` (std_msgs/Float32): Right thruster command

## Testing

Run unit tests:

```bash
cd ~/your_workspace
pytest src/modular_controller/tests/ -v
```

## Reference Frames

- **World/ENU Frame**: All guidance and control calculations
- **Body Frame**: Only for final thrust allocation

See `CONTROLLER_ARCHITECTURE.md` for detailed documentation.

## Dependencies

- rclpy
- nav_msgs
- geometry_msgs
- std_msgs
- jax
- scipy
- numpy
- pyyaml

## License

TODO
