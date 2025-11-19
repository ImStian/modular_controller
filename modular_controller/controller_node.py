import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64, Float64MultiArray
from geometry_msgs.msg import Pose2D
from nav_msgs.msg import Odometry
from modular_controller.los_guidance import LOSGuidance
from modular_controller.mrac import MRAC
from modular_controller.heading_controller import HeadingController, HeadingMode
from modular_controller.thrust_allocator import ThrustAllocator
from modular_controller.waypoint_path import load_waypoints_from_yaml
import os
import jax.numpy as jnp
from ament_index_python.packages import get_package_share_directory

class MainController(Node):
    def __init__(self):
        super().__init__('main_controller')

        # Declare parameters
        self.declare_parameter('waypoints_file', 'waypoints.yaml')
        self.declare_parameter('los_speed', 2.0)  
        self.declare_parameter('los_delta', 1.0) 
        self.declare_parameter('los_k', 0.3)   
        self.declare_parameter('heading_mode', 'los')  # 'los', 'path', or 'force'
        self.declare_parameter('k_psi', 40.0)  # Increased from 10.0 to prevent slingshot behavior
        self.declare_parameter('k_r', 15.0)    # Increased from 5.0 for better damping

        # Publishers for left and right thrusters (match simulator topic names)
        self.left_pub = self.create_publisher(Float64, '/model/blueboat/joint/motor_port_joint/cmd_thrust', 10)
        self.right_pub = self.create_publisher(Float64, '/model/blueboat/joint/motor_stbd_joint/cmd_thrust', 10)
        
        # Publisher for waypoints (for simulator visualization)
        self.waypoints_pub = self.create_publisher(Float64MultiArray, '/waypoints', 10)

        # Control loop timer (10 Hz)
        self.timer = self.create_timer(0.1, self.control_loop)

        # Load waypoint path from YAML
        waypoints_file = self.get_parameter('waypoints_file').get_parameter_value().string_value
        try:
            # Try package share directory first
            pkg_dir = get_package_share_directory('modular_controller')
            yaml_path = os.path.join(pkg_dir, 'config', waypoints_file)
        except:
            # Fall back to relative path
            yaml_path = os.path.join(
                os.path.dirname(__file__), '..', 'config', waypoints_file
            )
        
        self.get_logger().info(f"Loading waypoints from: {yaml_path}")
        self.path, self.path_settings = load_waypoints_from_yaml(yaml_path)
        self.get_logger().info(
            f"Loaded path with {len(self.path.get_waypoints())} waypoints, "
            f"total length: {self.path.get_path_length():.2f} m"
        )
        
        # Publish waypoints to simulator for visualization (immediately and after delay)
        self._publish_waypoints_to_simulator()
        
        # Republish after 1 second to ensure simulator has started
        self._waypoint_republish_timer = self.create_timer(1.0, self._publish_waypoints_to_simulator_once)

        # Controllers
        los_speed = self.get_parameter('los_speed').get_parameter_value().double_value
        los_delta = self.get_parameter('los_delta').get_parameter_value().double_value
        los_k = self.get_parameter('los_k').get_parameter_value().double_value
        
        self.los = LOSGuidance()
        self.los.los_parameters(U=los_speed, delta=los_delta, k=los_k)
        
        self.mrac = MRAC(
            tether_length=3.5,
            epsilon=0.7,
            k_v=1.5,
            k_a=1.5,
            logger=self.get_logger()
        )
        
        # Parse heading mode from parameter
        heading_mode_str = self.get_parameter('heading_mode').get_parameter_value().string_value
        if heading_mode_str.upper() == 'PATH':
            heading_mode = HeadingMode.PATH
        elif heading_mode_str.upper() == 'FORCE':
            heading_mode = HeadingMode.FORCE
        else:  # Default to LOS
            heading_mode = HeadingMode.LOS
        
        k_psi = self.get_parameter('k_psi').get_parameter_value().double_value
        k_r = self.get_parameter('k_r').get_parameter_value().double_value
        
        self.heading = HeadingController(
            k_psi=k_psi,
            k_r=k_r,
            mode=heading_mode,
            logger=self.get_logger()
        )
        self.thrust = ThrustAllocator()

        # ASV and towfish state (initialized to None to detect first message)
        self.asv_x = None
        self.asv_y = None
        self.asv_psi = None
        self.asv_r = None  # Yaw rate
        self.asv_vx = None  # Nav frame velocity
        self.asv_vy = None  # Nav frame velocity
        self.towfish_x = None
        self.towfish_y = None
        
        # Flag to track if we've received initial odometry
        self.received_asv_odom = False
        self.received_towfish_odom = False

        # Path variable (arc length parameter)
        self.s = 0.0
        
        # Store previous v_ref for finite difference
        self.v_ref_prev = jnp.array([0.0, 0.0])
        self.dt = 0.1  # Control loop period
        
        # Subscribers (using Odometry from simulator)
        self.create_subscription(Odometry, '/model/blueboat/odometry', self.asv_odom_callback, 10)
        self.create_subscription(Odometry, '/model/bluerov2_heavy/odometry', self.towfish_odom_callback, 10)

    def asv_odom_callback(self, msg: Odometry):
        """Process ASV odometry from simulator"""
        self.asv_x = msg.pose.pose.position.x
        self.asv_y = msg.pose.pose.position.y
        
        # Extract yaw from quaternion
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w
        self.asv_psi = 2.0 * jnp.arctan2(qz, qw)
        
        # Extract yaw rate from angular velocity
        self.asv_r = msg.twist.twist.angular.z
        
        # Get body-frame velocity and rotate to navigation frame
        vx_body = msg.twist.twist.linear.x
        vy_body = msg.twist.twist.linear.y
        
        c = jnp.cos(self.asv_psi)
        s = jnp.sin(self.asv_psi)
        self.asv_vx = c * vx_body - s * vy_body
        self.asv_vy = s * vx_body + c * vy_body
        
        self.received_asv_odom = True

    def towfish_odom_callback(self, msg: Odometry):
        """Process towfish odometry from simulator"""
        self.towfish_x = msg.pose.pose.position.x
        self.towfish_y = msg.pose.pose.position.y
        self.received_towfish_odom = True

    def control_loop(self):
        """Main control loop - runs at 10 Hz"""
        # Wait until we've received initial odometry from both vehicles
        if not self.received_asv_odom or not self.received_towfish_odom:
            self.get_logger().info("Waiting for odometry data...", throttle_duration_sec=1.0)
            return
        
        # Compute control point position: cp_pos = (1-epsilon)*asv_pos + epsilon*towfish_pos
        epsilon = self.mrac.epsilon
        asv_pos = [self.asv_x, self.asv_y]
        towfish_pos = [self.towfish_x, self.towfish_y]
        cp_pos = [
            (1 - epsilon) * asv_pos[0] + epsilon * towfish_pos[0],
            (1 - epsilon) * asv_pos[1] + epsilon * towfish_pos[1]
        ]
        
        # 1. LOS guidance - compute desired velocity and path rate using control point
        v_ref, s_dot = self.los.compute(
            position=cp_pos,
            s=self.s,
            path_function=self.path  # Use the waypoint path
        )
        
        # Update path parameter
        dt = 0.1  # 10 Hz control loop
        self.s += s_dot * dt
        self.s = max(0.0, min(self.s, self.path.get_path_length())) # Clamp s to path length
        

        # 2. Compute v_ref_dot using finite differences
        v_ref_dot = (v_ref - self.v_ref_prev) / self.dt
        self.v_ref_prev = v_ref

        # 3. MRAC -> Force Command
        asv_vel = jnp.array([self.asv_vx, self.asv_vy])
        
        u_p, zeta_dot = self.mrac.compute(
            asv_position=[self.asv_x, self.asv_y],
            asv_velocity=asv_vel,
            towfish_position=[self.towfish_x, self.towfish_y],
            v_ref=v_ref,
            v_ref_dot=v_ref_dot,
            dt=self.dt
        )
        # u_p is now in navigation frame [F_x, F_y]

        # 4. Heading controller - compute heading reference and yaw rate command
        if self.heading.mode == HeadingMode.PATH:
            psi_ref, r_ref = self.heading.compute_reference_from_path(
                s=self.s,
                s_dot=s_dot,
                path_function=self.path,
                dt=self.dt
            )
        elif self.heading.mode == HeadingMode.LOS:
            psi_ref, r_ref = self.heading.compute_reference_from_los(
                v_ref=v_ref,
                v_ref_dot=v_ref_dot
            )
        else:  # HeadingMode.FORCE
            psi_ref, r_ref = self.heading.compute_reference_from_force(u_p)
        
        # Compute heading control torque
        tau_r = self.heading.compute(
            psi=self.asv_psi,
            r=self.asv_r,
            psi_ref=psi_ref,
            r_ref=r_ref
        )

        # 5. Map force and torque to body frame for thrust allocation
        # Compute surge force in body frame
        F_surge = u_p[0] * jnp.cos(self.asv_psi) + u_p[1] * jnp.sin(self.asv_psi)
        
        # 6. Differential thrust allocation
        T_left, T_right = self.thrust.allocate(F_surge, tau_r)

        # 7. Publish directly to thruster topics
        self.left_pub.publish(Float64(data=float(T_left)))
        self.right_pub.publish(Float64(data=float(T_right)))

        # Logging
        self.get_logger().info(
            f"s={self.s:.2f}/{self.path.get_path_length():.2f}, "
            f"v_ref=[{v_ref[0]:.2f}, {v_ref[1]:.2f}], "
            f"v_ref_dot=[{v_ref_dot[0]:.2f}, {v_ref_dot[1]:.2f}], "
            f"u_p=[{u_p[0]:.2f}, {u_p[1]:.2f}], "
            f"psi_ref={psi_ref:.2f}, tau_r={tau_r:.2f}, "
            f"T_L={T_left:.2f}, T_R={T_right:.2f}"
        )
    
    def _publish_waypoints_to_simulator(self):
        """Publish interpolated path to simulator for smooth visualization."""
        import numpy as np
        
        # For smooth cubic/bspline paths, publish many interpolated points
        # For linear paths, just publish the waypoints
        path_length = self.path.get_path_length()
        
        # Sample the path at fine resolution for smooth display
        num_points = max(100, int(path_length * 2))  # 2 points per meter minimum
        s_values = np.linspace(0, path_length, num_points)
        
        # Get interpolated path points
        path_points = []
        for s in s_values:
            point = self.path(float(s))
            path_points.append(float(point[0]))
            path_points.append(float(point[1]))
        
        msg = Float64MultiArray()
        msg.data = path_points
        
        # Publish multiple times with small delay to ensure simulator receives it
        for _ in range(5):
            self.waypoints_pub.publish(msg)
        
        self.get_logger().info(
            f"Published interpolated path ({num_points} points over {path_length:.2f}m) "
            f"to simulator for visualization"
        )
    
    def _publish_waypoints_to_simulator_once(self):
        """One-time delayed republish of waypoints (for simulator startup timing)."""
        self._publish_waypoints_to_simulator()
        # Cancel this timer after first execution
        if hasattr(self, '_waypoint_republish_timer'):
            self._waypoint_republish_timer.cancel()

def main(args=None):
    rclpy.init(args=args)
    node = MainController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
