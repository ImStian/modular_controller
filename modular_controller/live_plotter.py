#!/usr/bin/env python3
"""
Live plotter for ASV and towfish positions from Gazebo simulator.
Subscribes to odometry topics and displays real-time trajectory visualization.
"""

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64MultiArray
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle
from collections import deque
import math


class LivePlotter(Node):
    def __init__(self):
        super().__init__('live_plotter')
        
        # Parameters
        self.declare_parameter('asv_odom_topic', '/model/blueboat/odometry')
        self.declare_parameter('towfish_odom_topic', '/model/bluerov2_heavy/odometry')
        self.declare_parameter('waypoints_topic', '/waypoints')
        self.declare_parameter('v_ref_topic', '/modular_controller/v_ref')  # New
        # Base history length (will be auto-expanded if needed)
        self.declare_parameter('history_length', 10000000)
        # Target seconds of trail retention; we will measure odom callback rate and
        # resize history deques to cover at least this duration.
        self.declare_parameter('desired_history_seconds', 360000)  # 1 hour default
        self.declare_parameter('update_rate_hz', 2)
        self.declare_parameter('show_tether', True)
        self.declare_parameter('show_boat_shape', True)
        self.declare_parameter('show_velocity_plot', True)  # New
        self.declare_parameter('tether_length', 3.5)  # MRAC parameter
        self.declare_parameter('epsilon', 0.7)  # MRAC parameter
        self.declare_parameter('save_plot_on_completion', True)  # Save plot when done
        self.declare_parameter('plot_save_path', 'tracking_plot.png')  # Save location
        
        asv_topic = self.get_parameter('asv_odom_topic').value
        towfish_topic = self.get_parameter('towfish_odom_topic').value
        waypoints_topic = self.get_parameter('waypoints_topic').value
        v_ref_topic = self.get_parameter('v_ref_topic').value
        history_len = self.get_parameter('history_length').value
        self.desired_history_seconds = self.get_parameter('desired_history_seconds').value
        update_rate = self.get_parameter('update_rate_hz').value
        self.show_tether = self.get_parameter('show_tether').value
        self.show_boat = self.get_parameter('show_boat_shape').value
        self.show_vel_plot = self.get_parameter('show_velocity_plot').value
        
        # MRAC parameters for tracking point calculation
        self.L = self.get_parameter('tether_length').value
        self.epsilon = self.get_parameter('epsilon').value
        self.save_on_complete = self.get_parameter('save_plot_on_completion').value
        self.plot_save_path = self.get_parameter('plot_save_path').value
        
        # State
        self.asv_pos = None
        self.asv_heading = None
        self.asv_vel = None
        self.towfish_pos = None
        self.towfish_vel = None  # New
        self.v_ref = None  # New
        self.path_points = None
        self.current_time = 0.0
        self.prev_theta = None  # For theta_dot calculation
        self.prev_time = None  # For dt calculation
        self.tracking_point_vel = None  # Computed tracking point velocity
        self.path_completed = False  # Track if we've reached the end of the path
        self.last_asv_pos = None  # For detecting path completion
        self.theta_dot_filtered = 0.0  # Filtered theta_dot to reduce noise
        self.path_length = 0.0  # Total path length
        self.distance_traveled = 0.0  # Distance traveled along path
        self.is_closed_loop = False  # Is this a closed loop path?
        
        # Track overall bounds for trajectory plot (not affected by deque limit)
        self.overall_x_min = float('inf')
        self.overall_x_max = float('-inf')
        self.overall_y_min = float('inf')
        self.overall_y_max = float('-inf')
        
        # History
        self.asv_history = deque(maxlen=history_len)
        self.towfish_history = deque(maxlen=history_len)
        self.time_history = deque(maxlen=history_len)  # New
        self.v_ref_history = deque(maxlen=history_len)  # New
        self.tracking_vel_history = deque(maxlen=history_len)  # Tracking point velocity

        # Callback rate measurement for dynamic history resizing
        self._asv_callback_times = deque(maxlen=5000)  # store recent timestamps
        self._rate_estimated = False
        self._dynamic_resize_done = False
        self._history_full_warned = False
        
        # Subscribers
        self.create_subscription(Odometry, asv_topic, self.asv_callback, 10)
        self.create_subscription(Odometry, towfish_topic, self.towfish_callback, 10)
        self.create_subscription(Float64MultiArray, waypoints_topic, self.waypoints_callback, 10)
        self.create_subscription(Float64MultiArray, v_ref_topic, self.v_ref_callback, 10)  # New
        
        # Setup matplotlib with subplots
        plt.ion()
        if self.show_vel_plot:
            self.fig = plt.figure(figsize=(16, 10))
            # 2x2 grid: trajectory takes entire first row, velocity plots on second row
            self.ax_traj = plt.subplot2grid((2, 2), (0, 0), colspan=2)
            self.ax_vel_y = plt.subplot2grid((2, 2), (1, 0))  # North velocity (left)
            self.ax_vel_x = plt.subplot2grid((2, 2), (1, 1))  # East velocity (right)
        else:
            self.fig, self.ax_traj = plt.subplots(figsize=(12, 10))
            self.ax_vel_x = None
            self.ax_vel_y = None
        
        # === TRAJECTORY PLOT ===
        self.path_line, = self.ax_traj.plot([], [], 'k--', lw=1.0, label='Reference Path', alpha=0.5)
        self.asv_trail, = self.ax_traj.plot([], [], 'b-', lw=1.5, label='ASV Trail', alpha=0.7)
        self.towfish_trail, = self.ax_traj.plot([], [], 'r-', lw=1.5, label='Towfish Trail', alpha=0.7)
        self.tether_line, = self.ax_traj.plot([], [], 'gray', lw=2, linestyle=':', alpha=0.6, label='Tether')
        
        # ASV boat shape
        if self.show_boat:
            boat_length = 0.8
            boat_width = 0.4
            boat_shape = np.array([
                [boat_length, 0.0],
                [-boat_length/3, boat_width/2],
                [-boat_length/3, -boat_width/2],
            ])
            self.asv_boat = Polygon(boat_shape, closed=True, facecolor='blue', 
                                   edgecolor='darkblue', linewidth=2, alpha=0.8)
            self.ax_traj.add_patch(self.asv_boat)
        else:
            self.asv_boat = None
            
        # Current positions
        self.asv_marker = self.ax_traj.scatter([], [], c='blue', marker='o', s=150, 
                                         edgecolors='darkblue', linewidths=2, 
                                         label='ASV', zorder=10)
        self.towfish_marker = self.ax_traj.scatter([], [], c='red', marker='x', s=200, 
                                             linewidths=3, label='Towfish', zorder=10)
        
        # Trajectory plot settings
        self.ax_traj.set_xlabel('East [m]', fontsize=12)
        self.ax_traj.set_ylabel('North [m]', fontsize=12)
        self.ax_traj.set_title('ASV-Towfish Live Tracking', fontsize=14, fontweight='bold')
        self.ax_traj.grid(True, alpha=0.3)
        self.ax_traj.set_ylim(-15, 15)
        self.ax_traj.set_xlim(-5, 75)
        #self.ax_traj.set_aspect('equal', 'box')
        self.ax_traj.legend(loc='upper right', fontsize=9)
        
        # === VELOCITY PLOTS ===
        if self.show_vel_plot:
            # Y-velocity plot (North - left column)
            self.v_ref_y_line, = self.ax_vel_y.plot([], [], 'g-', lw=2, label='v_ref_y (LOS)', alpha=0.8)
            self.tracking_vy_line, = self.ax_vel_y.plot([], [], 'r-', lw=2, label='Tracking Point v_y', alpha=0.8)
            self.ax_vel_y.set_xlabel('Time [s]', fontsize=10)
            self.ax_vel_y.set_ylabel('Y Velocity [m/s]', fontsize=10)
            self.ax_vel_y.set_title('North Velocity Tracking', fontsize=11, fontweight='bold')
            self.ax_vel_y.grid(True, alpha=0.3)
            self.ax_vel_y.set_ylim(-2, 2)  # Initialize limits
            self.ax_vel_y.yaxis.set_major_locator(plt.MultipleLocator(0.2))  # 0.2 spacing
            self.ax_vel_y.legend(loc='upper right', fontsize=8)
            
            # X-velocity plot (East - right column)
            self.v_ref_x_line, = self.ax_vel_x.plot([], [], 'g-', lw=2, label='v_ref_x (LOS)', alpha=0.8)
            self.tracking_vx_line, = self.ax_vel_x.plot([], [], 'r-', lw=2, label='Tracking Point v_x', alpha=0.8)
            self.ax_vel_x.set_xlabel('Time [s]', fontsize=10)
            self.ax_vel_x.set_ylabel('X Velocity [m/s]', fontsize=10)
            self.ax_vel_x.set_title('East Velocity Tracking', fontsize=11, fontweight='bold')
            self.ax_vel_x.grid(True, alpha=0.3)
            self.ax_vel_x.set_ylim(-2, 2)  # Initialize limits
            self.ax_vel_x.yaxis.set_major_locator(plt.MultipleLocator(0.2))  # 0.2 spacing
            self.ax_vel_x.legend(loc='upper right', fontsize=8)
        
        plt.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
        # Update timer
        self.plot_timer = self.create_timer(1.0 / update_rate, self.update_plot)
        
        self.get_logger().info(f'Live plotter started')
        self.get_logger().info(f'  ASV topic: {asv_topic}')
        self.get_logger().info(f'  Towfish topic: {towfish_topic}')
        self.get_logger().info(f'  Waypoints topic: {waypoints_topic}')
        self.get_logger().info(f'  V_ref topic: {v_ref_topic}')
        
    def asv_callback(self, msg: Odometry):
        """Process ASV odometry"""
        self.asv_pos = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y])

        # Record callback time for rate estimation
        now_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self._asv_callback_times.append(now_time)
        if not self._rate_estimated and len(self._asv_callback_times) > 1000:
            # Estimate mean dt over collected samples
            total_dt = self._asv_callback_times[-1] - self._asv_callback_times[0]
            sample_count = len(self._asv_callback_times) - 1
            if total_dt > 0 and sample_count > 0:
                mean_dt = total_dt / sample_count
                est_rate = 1.0 / mean_dt
                self._rate_estimated = True
                self.get_logger().info(f'Estimated ASV odom rate: {est_rate:.1f} Hz')
                # Compute required length
                required_len = int(est_rate * self.desired_history_seconds * 1.1)  # 10% margin
                if required_len > self.asv_history.maxlen:
                    self.get_logger().info(f'Resizing history deques to {required_len} (covers ~{self.desired_history_seconds}s)')
                    # Rebuild deques with larger capacity
                    self.asv_history = deque(self.asv_history, maxlen=required_len)
                    self.towfish_history = deque(self.towfish_history, maxlen=required_len)
                    self.time_history = deque(self.time_history, maxlen=required_len)
                    self.v_ref_history = deque(self.v_ref_history, maxlen=required_len)
                    self.tracking_vel_history = deque(self.tracking_vel_history, maxlen=required_len)
                    self._dynamic_resize_done = True
                else:
                    self.get_logger().info('Existing history_length sufficient for desired retention.')
        
        # Track overall bounds for trajectory plot
        self.overall_x_min = min(self.overall_x_min, self.asv_pos[0])
        self.overall_x_max = max(self.overall_x_max, self.asv_pos[0])
        self.overall_y_min = min(self.overall_y_min, self.asv_pos[1])
        self.overall_y_max = max(self.overall_y_max, self.asv_pos[1])
        
        # Track distance traveled
        if self.last_asv_pos is not None:
            self.distance_traveled += np.linalg.norm(self.asv_pos - self.last_asv_pos)
        self.last_asv_pos = self.asv_pos.copy()
        
        # Extract yaw from quaternion
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w
        self.asv_heading = 2.0 * math.atan2(qz, qw)
        
        # Extract ASV velocity in BODY frame and transform to NAVIGATION frame
        vx_body = msg.twist.twist.linear.x
        vy_body = msg.twist.twist.linear.y
        
        # Rotate from body frame to navigation frame
        c = np.cos(self.asv_heading)
        s = np.sin(self.asv_heading)
        vx_nav = c * vx_body - s * vy_body
        vy_nav = s * vx_body + c * vy_body
        
        self.asv_vel = np.array([vx_nav, vy_nav])  # Now in navigation frame!
        
        self.asv_history.append(self.asv_pos.copy())
        # Warn once if history gets full and we didn't resize (user can increase param)
        if not self._history_full_warned and len(self.asv_history) == self.asv_history.maxlen:
            self._history_full_warned = True
            duration_covered = len(self.asv_history) / (len(self._asv_callback_times) / (self._asv_callback_times[-1] - self._asv_callback_times[0] + 1e-6)) if len(self._asv_callback_times) > 10 else 0
            self.get_logger().warn(
                f'Trail history full at {len(self.asv_history)} points; earlier path will drop. '
                f'Increase history_length or desired_history_seconds (currently {self.desired_history_seconds}) if full-path retention needed.'
            )
        
        # Compute tracking point velocity (same method as MRAC)
        self._compute_tracking_point_velocity(msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9)
        
    def towfish_callback(self, msg: Odometry):
        """Process towfish odometry"""
        self.towfish_pos = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y])
        
        # Track overall bounds for trajectory plot
        self.overall_x_min = min(self.overall_x_min, self.towfish_pos[0])
        self.overall_x_max = max(self.overall_x_max, self.towfish_pos[0])
        self.overall_y_min = min(self.overall_y_min, self.towfish_pos[1])
        self.overall_y_max = max(self.overall_y_max, self.towfish_pos[1])
        
        # Extract velocity in navigation frame (for reference, not used in tracking)
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        self.towfish_vel = np.array([vx, vy])
        
        self.towfish_history.append(self.towfish_pos.copy())
        
        # Update time
        self.current_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.time_history.append(self.current_time)
        
    def _compute_tracking_point_velocity(self, current_time):
        """
        Compute tracking point velocity using MRAC's pendulum model.
        This is the point at distance epsilon*L from ASV along the tether.
        
        Based on MRAC.compute() method:
        v = asv_vel + epsilon * L * theta_dot * dGamma
        """
        if self.asv_pos is None or self.towfish_pos is None or self.asv_vel is None:
            return
            
        # Compute pendulum angle from relative positions
        dx = self.towfish_pos - self.asv_pos
        distance = np.linalg.norm(dx)
        
        if distance < 1e-6:
            self.tracking_point_vel = self.asv_vel.copy()
            self.tracking_vel_history.append(self.tracking_point_vel.copy())
            return
            
        # Pendulum angle (theta) in navigation frame
        theta = np.arctan2(dx[1], dx[0])
        
        # Compute theta_dot using finite differences
        if self.prev_theta is not None and self.prev_time is not None:
            dt = current_time - self.prev_time
            if dt > 1e-6:
                dtheta = theta - self.prev_theta
                # Wrap angle difference to [-pi, pi]
                dtheta = (dtheta + np.pi) % (2 * np.pi) - np.pi
                theta_dot_raw = dtheta / dt
                
                # Apply low-pass filter to reduce noise (exponential moving average)
                # alpha = 0.1 means 10% new value, 90% old value (strong smoothing)
                alpha = 0.1
                self.theta_dot_filtered = alpha * theta_dot_raw + (1 - alpha) * self.theta_dot_filtered
                theta_dot = self.theta_dot_filtered
            else:
                theta_dot = self.theta_dot_filtered
        else:
            theta_dot = 0.0
            self.theta_dot_filtered = 0.0
            
        # Update previous values
        self.prev_theta = theta
        self.prev_time = current_time
        
        # Pendulum basis vector derivative (perpendicular to radial direction)
        dGamma = np.array([-np.sin(theta), np.cos(theta)])
        
        # Tracking point velocity (MRAC's tracked point)
        # v = asv_vel + epsilon * L * theta_dot * dGamma
        self.tracking_point_vel = self.asv_vel + self.epsilon * self.L * theta_dot * dGamma
        
        # Store in history
        self.tracking_vel_history.append(self.tracking_point_vel.copy())
        
    def v_ref_callback(self, msg: Float64MultiArray):
        """Process reference velocity from controller"""
        if len(msg.data) >= 2:
            self.v_ref = np.array([msg.data[0], msg.data[1]])
            self.v_ref_history.append(self.v_ref.copy())
        
    def waypoints_callback(self, msg: Float64MultiArray):
        """Process waypoints for path visualization"""
        data = list(msg.data)
        if len(data) >= 4:
            pts = [(data[i], data[i+1]) for i in range(0, len(data), 2)]
            self.path_points = np.array(pts, dtype=float)
            
            # Update overall bounds with waypoints
            if len(self.path_points) > 0:
                self.overall_x_min = min(self.overall_x_min, np.min(self.path_points[:, 0]))
                self.overall_x_max = max(self.overall_x_max, np.max(self.path_points[:, 0]))
                self.overall_y_min = min(self.overall_y_min, np.min(self.path_points[:, 1]))
                self.overall_y_max = max(self.overall_y_max, np.max(self.path_points[:, 1]))
            
            # Compute total path length
            if len(self.path_points) > 1:
                segments = np.diff(self.path_points, axis=0)
                segment_lengths = np.linalg.norm(segments, axis=1)
                self.path_length = np.sum(segment_lengths)
                
                # Check if closed loop (start and end are very close)
                start_end_dist = np.linalg.norm(self.path_points[0] - self.path_points[-1])
                self.is_closed_loop = start_end_dist < 0.5  # Within 0.5m
                
                self.get_logger().info(f'Path length: {self.path_length:.2f}m, Closed loop: {self.is_closed_loop}')
            
    def _check_path_completion(self):
        """Check if ASV has completed the path"""
        if self.path_points is None or len(self.path_points) == 0:
            return False
            
        if self.asv_pos is None:
            return False
        
        # For closed loop paths (like circle), check if traveled full distance
        if self.is_closed_loop and self.path_length > 0:
            completion_ratio = self.distance_traveled / self.path_length
            # Also check if back near starting point
            start_pos = self.path_points[0]
            dist_to_start = np.linalg.norm(self.asv_pos - start_pos)
            
            if completion_ratio >= 0.999 and dist_to_start < 4.0:
                self.get_logger().info(f'Path complete: traveled {self.distance_traveled:.2f}m / {self.path_length:.2f}m, dist to start: {dist_to_start:.2f}m')
                return True
        
        # For open paths, check x-position threshold
        if (not self.is_closed_loop) and (self.asv_pos[0] > 70.0): # Adjust this depending on when you want the plot to be saved
            self.get_logger().info(f'Path complete: ASV x={self.asv_pos[0]:.2f}m (threshold: 70m)')
            return True
        
        return False
    
    def _save_plot(self):
        """Save the current plot to file"""
        if self.path_completed:
            return  # Already saved
            
        try:
            self.get_logger().info(f'Saving plot to {self.plot_save_path}')
            self.fig.savefig(self.plot_save_path, dpi=300, bbox_inches='tight')
            self.get_logger().info(f'Plot saved successfully!')
            self.path_completed = True
            # Prevent re-triggering by setting distance_traveled >= path_length
            if self.path_length > 0:
                self.distance_traveled = self.path_length
        except Exception as e:
            self.get_logger().error(f'Failed to save plot: {e}')
            
    def update_plot(self):
        """Update the plot with current data"""
        try:
            # === UPDATE TRAJECTORY PLOT ===
            # Update reference path
            if self.path_points is not None and len(self.path_points) > 1:
                self.path_line.set_data(self.path_points[:, 0], self.path_points[:, 1])
            
            # Update trails
            if len(self.asv_history) > 0:
                asv_trail = np.array(self.asv_history)
                self.asv_trail.set_data(asv_trail[:, 0], asv_trail[:, 1])
                
            if len(self.towfish_history) > 0:
                towfish_trail = np.array(self.towfish_history)
                self.towfish_trail.set_data(towfish_trail[:, 0], towfish_trail[:, 1])
            
            # Update current positions
            if self.asv_pos is not None:
                self.asv_marker.set_offsets([self.asv_pos])
                
                # Update boat shape
                if self.show_boat and self.asv_boat is not None and self.asv_heading is not None:
                    boat_length = 0.8
                    boat_width = 0.4
                    boat_local = np.array([
                        [boat_length, 0.0],
                        [-boat_length/3, boat_width/2],
                        [-boat_length/3, -boat_width/2],
                    ])
                    # Rotation matrix
                    c, s = np.cos(self.asv_heading), np.sin(self.asv_heading)
                    R = np.array([[c, -s], [s, c]])
                    boat_rotated = (R @ boat_local.T).T
                    boat_world = boat_rotated + self.asv_pos
                    self.asv_boat.set_xy(boat_world)
                    
            if self.towfish_pos is not None:
                self.towfish_marker.set_offsets([self.towfish_pos])
                
            # Update tether line
            if self.show_tether and self.asv_pos is not None and self.towfish_pos is not None:
                self.tether_line.set_data(
                    [self.asv_pos[0], self.towfish_pos[0]], 
                    [self.asv_pos[1], self.towfish_pos[1]]
                )
            
            # Auto-scale trajectory view using overall bounds (not limited by deque size)
            if (self.overall_x_min != float('inf') and self.overall_x_max != float('-inf') and
                self.overall_y_min != float('inf') and self.overall_y_max != float('-inf')):
                margin = 2.0  # meters
                self.ax_traj.set_xlim(self.overall_x_min - margin, self.overall_x_max + margin)
                # Keep y-axis fixed or uncomment to auto-scale:
                # self.ax_traj.set_ylim(self.overall_y_min - margin, self.overall_y_max + margin)
            
            # === UPDATE VELOCITY PLOTS ===
            if self.show_vel_plot and len(self.time_history) > 1:
                # Sync histories (trim to same length)
                min_len = min(len(self.time_history), len(self.v_ref_history), len(self.tracking_vel_history))
                
                if min_len > 1:
                    # Get time vector (relative to start)
                    times = np.array(list(self.time_history))
                    t0 = times[0]
                    t_rel = times - t0
                    
                    # Get velocity data
                    v_refs = np.array(list(self.v_ref_history))
                    tracking_vels = np.array(list(self.tracking_vel_history))
                    
                    # Trim to same length
                    t_rel = t_rel[-min_len:]
                    v_refs = v_refs[-min_len:]
                    tracking_vels = tracking_vels[-min_len:]
                    
                    # Update X-velocity plot
                    self.v_ref_x_line.set_data(t_rel, v_refs[:, 0])
                    self.tracking_vx_line.set_data(t_rel, tracking_vels[:, 0])
                    # Only auto-scale x-axis (time), keep y-axis fixed at [-2, 2]
                    self.ax_vel_x.relim()
                    self.ax_vel_x.autoscale_view(scalex=True, scaley=False)
                    
                    # Update Y-velocity plot
                    self.v_ref_y_line.set_data(t_rel, v_refs[:, 1])
                    self.tracking_vy_line.set_data(t_rel, tracking_vels[:, 1])
                    # Only auto-scale x-axis (time), keep y-axis fixed at [-2, 2]
                    self.ax_vel_y.relim()
                    self.ax_vel_y.autoscale_view(scalex=True, scaley=False)
            
            # Check for path completion and save plot
            if self.save_on_complete and not self.path_completed:
                if self._check_path_completion():
                    self._save_plot()
            
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            
        except Exception as e:
            self.get_logger().error(f'Plot update error: {e}')


def main(args=None):
    rclpy.init(args=args)
    plotter = LivePlotter()
    
    try:
        rclpy.spin(plotter)
    except KeyboardInterrupt:
        pass
    finally:
        plotter.destroy_node()
        rclpy.shutdown()
        plt.close('all')


if __name__ == '__main__':
    main()
