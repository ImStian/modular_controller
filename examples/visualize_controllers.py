"""
Visualization script for modular controller components.
Generates plots based on test scenarios to verify controller behavior.
"""
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import sys
import os

# Add modular_controller to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from modular_controller.mrac import MRAC
from modular_controller.heading_controller import HeadingController, HeadingMode
from modular_controller.thrust_allocator import ThrustAllocator
from modular_controller.los_guidance import LOSGuidance
from modular_controller.waypoint_path import WaypointPath


def plot_mrac_adaptation():
    """Plot MRAC adaptive parameter evolution and tracking performance."""
    print("Generating MRAC adaptation plots...")
    
    mrac = MRAC(tether_length=3.5, epsilon=0.7, k_v=1.5, k_a=1.5)
    
    # Simulation parameters
    dt = 0.1
    t_sim = 20.0
    steps = int(t_sim / dt)
    time = np.linspace(0, t_sim, steps)
    
    # Initialize states
    asv_pos = np.array([0.0, 0.0])
    asv_vel = np.array([0.0, 0.0])
    towfish_pos = np.array([3.5, 0.0])
    towfish_vel = np.array([0.0, 0.0])
    
    # Reference trajectory: constant velocity with acceleration phase
    v_ref_target = np.array([1.5, 0.0])
    
    # Storage
    forces = []
    zetas = []
    towfish_velocities = []
    tracking_errors = []
    v_refs = []
    
    for i in range(steps):
        # Ramp up reference velocity
        if time[i] < 2.0:
            v_ref = v_ref_target * (time[i] / 2.0)
            v_ref_dot = v_ref_target / 2.0
        else:
            v_ref = v_ref_target
            v_ref_dot = jnp.array([0.0, 0.0])
        
        # MRAC control
        u_p, zeta_dot = mrac.compute(
            asv_position=asv_pos.tolist(),
            asv_velocity=asv_vel.tolist(),
            towfish_position=towfish_pos.tolist(),
            v_ref=v_ref,
            v_ref_dot=v_ref_dot,
            dt=dt
        )
        
        # Update adaptive parameters
        mrac.zeta += zeta_dot * dt
        
        # Simple dynamics integration (for visualization)
        # Towfish: m * v_dot = u_p + drag
        m_towfish = 25.0
        drag_coeff = 5.0
        towfish_acc = (np.array(u_p) - drag_coeff * towfish_vel) / m_towfish
        towfish_vel += towfish_acc * dt
        towfish_pos += towfish_vel * dt
        
        # ASV follows roughly (simplified)
        asv_vel = towfish_vel * 1.1
        asv_pos += asv_vel * dt
        
        # Store data
        forces.append(np.array(u_p))
        zetas.append(np.array(mrac.zeta))
        towfish_velocities.append(towfish_vel.copy())
        tracking_errors.append(np.linalg.norm(towfish_vel - np.array(v_ref)))
        v_refs.append(np.array(v_ref))
    
    forces = np.array(forces)
    zetas = np.array(zetas)
    towfish_velocities = np.array(towfish_velocities)
    tracking_errors = np.array(tracking_errors)
    v_refs = np.array(v_refs)
    
    # Create figure
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(3, 2, figure=fig)
    
    # Plot 1: Velocity tracking
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(time, v_refs[:, 0], 'k--', label='v_ref_x', linewidth=2)
    ax1.plot(time, towfish_velocities[:, 0], 'b-', label='v_towfish_x')
    ax1.plot(time, v_refs[:, 1], 'k:', label='v_ref_y', linewidth=2)
    ax1.plot(time, towfish_velocities[:, 1], 'r-', label='v_towfish_y')
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Velocity [m/s]')
    ax1.set_title('MRAC Velocity Tracking Performance')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Control forces
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(time, forces[:, 0], 'b-', label='F_x')
    ax2.plot(time, forces[:, 1], 'r-', label='F_y')
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Force [N]')
    ax2.set_title('MRAC Control Forces')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Tracking error
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(time, tracking_errors, 'g-', linewidth=2)
    ax3.set_xlabel('Time [s]')
    ax3.set_ylabel('Tracking Error [m/s]')
    ax3.set_title('Velocity Tracking Error Norm')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Adaptive parameters evolution (selected)
    ax4 = fig.add_subplot(gs[2, :])
    for i in range(min(5, zetas.shape[1])):
        ax4.plot(time, zetas[:, i], label=f'ζ_{i}')
    ax4.set_xlabel('Time [s]')
    ax4.set_ylabel('Parameter Value')
    ax4.set_title('MRAC Adaptive Parameters Evolution (first 5)')
    ax4.legend(ncol=5)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('mrac_adaptation.png', dpi=150, bbox_inches='tight')
    print("  Saved: mrac_adaptation.png")
    plt.close()


def plot_heading_controller_modes():
    """Plot heading controller performance in different modes."""
    print("Generating heading controller mode comparison plots...")
    
    # Simulation parameters
    dt = 0.1
    t_sim = 15.0
    steps = int(t_sim / dt)
    time = np.linspace(0, t_sim, steps)
    
    # Create SHARED sinusoidal path for all modes
    waypoints = []
    for x in np.linspace(0, 60, 13):
        y = 10 * np.sin(x / 10.0)
        waypoints.append([float(x), float(y)])
    
    shared_path = WaypointPath(waypoints, method='cubic')
    
    # Shared controllers
    shared_los = LOSGuidance()
    shared_los.los_parameters(U=2.0, delta=5.0, k=0.15)
    
    shared_mrac = MRAC(tether_length=3.5, epsilon=0.7, k_v=1.5, k_a=1.5)
    
    modes = [HeadingMode.PATH, HeadingMode.LOS, HeadingMode.FORCE]
    mode_names = ['PATH', 'LOS', 'FORCE']
    colors = ['blue', 'green', 'red']
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    
    for mode_idx, (mode, mode_name, color) in enumerate(zip(modes, mode_names, colors)):
        controller = HeadingController(k_psi=10.0, k_r=5.0, mode=mode)
        
        # Reset MRAC for each mode
        shared_mrac.reset()
        
        # Initialize vessel state - SAME for all modes
        psi = 0.5  # Same initial heading error
        r = 0.0    # Initial yaw rate
        
        # Initialize positions - start at s=1.0 to avoid path gradient issues
        s = 1.0
        start_pos = shared_path(s)
        towfish_pos = np.array([float(start_pos[0]), float(start_pos[1]) - 3.0])  # Same off-path position
        towfish_vel = np.array([0.0, 0.0])
        asv_pos = towfish_pos - 3.5 * np.array([np.cos(psi), np.sin(psi)])
        asv_vel = np.array([0.0, 0.0])
        
        # Storage
        headings = []
        yaw_rates = []
        torques = []
        heading_refs = []
        yaw_rate_refs = []
        heading_errors = []
        v_ref_prev = jnp.array([0.0, 0.0])
        
        for i in range(steps):
            # Compute LOS guidance (same for all modes)
            v_ref, s_dot = shared_los.compute(
                position=towfish_pos.tolist(),
                s=s,
                path_function=shared_path
            )
            
            # Compute v_ref_dot
            if i > 0:
                v_ref_dot = (v_ref - v_ref_prev) / dt
            else:
                v_ref_dot = jnp.array([0.0, 0.0])
            v_ref_prev = v_ref
            
            # Compute MRAC control force (same for all modes)
            u_p, zeta_dot = shared_mrac.compute(
                asv_position=asv_pos.tolist(),
                asv_velocity=asv_vel.tolist(),
                towfish_position=towfish_pos.tolist(),
                v_ref=v_ref,
                v_ref_dot=v_ref_dot,
                dt=dt
            )
            shared_mrac.zeta += zeta_dot * dt
            
            # NOW compute different heading references from SAME inputs
            if mode == HeadingMode.PATH:
                psi_ref, r_ref = controller.compute_reference_from_path(
                    s=s, s_dot=s_dot, path_function=shared_path, dt=dt
                )
            elif mode == HeadingMode.LOS:
                psi_ref, r_ref = controller.compute_reference_from_los(v_ref, v_ref_dot)
            else:  # FORCE mode
                psi_ref, r_ref = controller.compute_reference_from_force(u_p)
            
            # Compute control torque
            tau_r = controller.compute(psi, r, psi_ref, r_ref)
            
            # Transform MRAC force to body frame using ACTUAL heading
            u_p_body_x = u_p[0] * np.cos(psi) + u_p[1] * np.sin(psi)
            u_p_body_y = -u_p[0] * np.sin(psi) + u_p[1] * np.cos(psi)
            
            # ASV dynamics with coupled heading
            m_asv = 50.0
            I = 8.0
            d_surge = 10.0
            
            # Yaw dynamics
            r_dot = (tau_r - 2.0 * r) / I
            r += r_dot * dt
            psi += r * dt
            psi = (psi + np.pi) % (2 * np.pi) - np.pi
            
            # Surge velocity in body frame (simplified, affected by heading alignment)
            u_body = 2.0  # Target surge speed
            
            # Transform to navigation frame
            asv_vel = u_body * np.array([np.cos(psi), np.sin(psi)])
            asv_pos += asv_vel * dt
            
            # Towfish dynamics
            m_towfish = 25.0
            drag = 5.0
            towfish_acc = (np.array(u_p) - drag * towfish_vel) / m_towfish
            towfish_vel += towfish_acc * dt
            towfish_pos += towfish_vel * dt
            
            # Update path parameter
            s += s_dot * dt
            s = min(s, shared_path.get_path_length())
            
            # Store
            headings.append(psi)
            yaw_rates.append(r)
            torques.append(float(tau_r))
            heading_refs.append(float(psi_ref))
            yaw_rate_refs.append(float(r_ref))
            heading_errors.append(float(psi - psi_ref))
        
        headings = np.array(headings)
        yaw_rates = np.array(yaw_rates)
        torques = np.array(torques)
        heading_refs = np.array(heading_refs)
        yaw_rate_refs = np.array(yaw_rate_refs)
        heading_errors = np.array(heading_errors)
        
        # Plot heading tracking
        axes[0, mode_idx].plot(time, np.rad2deg(heading_refs), 'k--', 
                               label='Reference', linewidth=2)
        axes[0, mode_idx].plot(time, np.rad2deg(headings), color, 
                               label='Actual', linewidth=1.5)
        axes[0, mode_idx].set_ylabel('Heading [deg]')
        axes[0, mode_idx].set_title(f'{mode_name} Mode: Heading')
        axes[0, mode_idx].legend()
        axes[0, mode_idx].grid(True, alpha=0.3)
        
        # Plot yaw rate
        axes[1, mode_idx].plot(time, yaw_rate_refs, 'k--', 
                               label='Reference', linewidth=2)
        axes[1, mode_idx].plot(time, yaw_rates, color, 
                               label='Actual', linewidth=1.5)
        axes[1, mode_idx].set_ylabel('Yaw Rate [rad/s]')
        axes[1, mode_idx].set_title(f'{mode_name} Mode: Yaw Rate')
        axes[1, mode_idx].legend()
        axes[1, mode_idx].grid(True, alpha=0.3)
        
        # Plot control torque
        axes[2, mode_idx].plot(time, torques, color, linewidth=1.5)
        axes[2, mode_idx].set_xlabel('Time [s]')
        axes[2, mode_idx].set_ylabel('Torque [N⋅m]')
        axes[2, mode_idx].set_title(f'{mode_name} Mode: Control Torque')
        axes[2, mode_idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('heading_controller_modes.png', dpi=150, bbox_inches='tight')
    print("  Saved: heading_controller_modes.png")
    plt.close()


def plot_thrust_allocation():
    """Plot thrust allocation for various scenarios."""
    print("Generating thrust allocation plots...")
    
    allocator = ThrustAllocator(thruster_separation=0.5, max_thrust=50.0)
    
    # Create test scenarios
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Scenario 1: Pure surge force
    F_surge_range = np.linspace(-60, 60, 100)
    T_left_1 = []
    T_right_1 = []
    for F in F_surge_range:
        T_l, T_r = allocator.allocate(F, 0.0)
        T_left_1.append(T_l)
        T_right_1.append(T_r)
    
    axes[0, 0].plot(F_surge_range, T_left_1, 'b-', label='T_left', linewidth=2)
    axes[0, 0].plot(F_surge_range, T_right_1, 'r-', label='T_right', linewidth=2)
    axes[0, 0].axhline(50, color='k', linestyle='--', alpha=0.5, label='Max thrust')
    axes[0, 0].axhline(-50, color='k', linestyle='--', alpha=0.5)
    axes[0, 0].set_xlabel('Surge Force [N]')
    axes[0, 0].set_ylabel('Thrust [N]')
    axes[0, 0].set_title('Pure Surge Force (τ_r = 0)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Scenario 2: Pure yaw torque
    tau_r_range = np.linspace(-30, 30, 100)
    T_left_2 = []
    T_right_2 = []
    for tau in tau_r_range:
        T_l, T_r = allocator.allocate(0.0, tau)
        T_left_2.append(T_l)
        T_right_2.append(T_r)
    
    axes[0, 1].plot(tau_r_range, T_left_2, 'b-', label='T_left', linewidth=2)
    axes[0, 1].plot(tau_r_range, T_right_2, 'r-', label='T_right', linewidth=2)
    axes[0, 1].axhline(50, color='k', linestyle='--', alpha=0.5, label='Max thrust')
    axes[0, 1].axhline(-50, color='k', linestyle='--', alpha=0.5)
    axes[0, 1].set_xlabel('Yaw Torque [N⋅m]')
    axes[0, 1].set_ylabel('Thrust [N]')
    axes[0, 1].set_title('Pure Yaw Torque (F_surge = 0)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Scenario 3: Combined allocation heatmap
    F_surge_grid = np.linspace(-40, 60, 50)
    tau_r_grid = np.linspace(-20, 20, 50)
    F_grid, Tau_grid = np.meshgrid(F_surge_grid, tau_r_grid)
    
    T_left_grid = np.zeros_like(F_grid)
    T_right_grid = np.zeros_like(F_grid)
    saturated = np.zeros_like(F_grid, dtype=bool)
    
    for i in range(len(tau_r_grid)):
        for j in range(len(F_surge_grid)):
            T_l, T_r = allocator.allocate(F_grid[i, j], Tau_grid[i, j])
            T_left_grid[i, j] = T_l
            T_right_grid[i, j] = T_r
            saturated[i, j] = (abs(T_l) >= 49.9) or (abs(T_r) >= 49.9)
    
    im = axes[0, 2].contourf(F_grid, Tau_grid, saturated.astype(int), 
                             levels=[0, 0.5, 1], colors=['lightgreen', 'salmon'],
                             alpha=0.7)
    axes[0, 2].contour(F_grid, Tau_grid, saturated.astype(int), 
                       levels=[0.5], colors='red', linewidths=2)
    axes[0, 2].set_xlabel('Surge Force [N]')
    axes[0, 2].set_ylabel('Yaw Torque [N⋅m]')
    axes[0, 2].set_title('Saturation Map (Red = Saturated)')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Scenario 4: Time-varying maneuver
    t_maneuver = np.linspace(0, 10, 100)
    F_surge_t = 30 * np.sin(0.5 * t_maneuver)
    tau_r_t = 15 * np.cos(0.8 * t_maneuver)
    
    T_left_t = []
    T_right_t = []
    for F, tau in zip(F_surge_t, tau_r_t):
        T_l, T_r = allocator.allocate(F, tau)
        T_left_t.append(T_l)
        T_right_t.append(T_r)
    
    axes[1, 0].plot(t_maneuver, T_left_t, 'b-', label='T_left', linewidth=2)
    axes[1, 0].plot(t_maneuver, T_right_t, 'r-', label='T_right', linewidth=2)
    axes[1, 0].axhline(50, color='k', linestyle='--', alpha=0.5)
    axes[1, 0].axhline(-50, color='k', linestyle='--', alpha=0.5)
    axes[1, 0].set_xlabel('Time [s]')
    axes[1, 0].set_ylabel('Thrust [N]')
    axes[1, 0].set_title('Dynamic Maneuver')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Scenario 5: Allocation efficiency
    axes[1, 1].plot(t_maneuver, F_surge_t, 'g-', label='F_surge', linewidth=2)
    axes[1, 1].plot(t_maneuver, tau_r_t, 'm-', label='τ_r', linewidth=2)
    axes[1, 1].set_xlabel('Time [s]')
    axes[1, 1].set_ylabel('Command')
    axes[1, 1].set_title('Input Commands')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Scenario 6: Thrust differential
    T_diff = np.array(T_right_t) - np.array(T_left_t)
    T_sum = np.array(T_right_t) + np.array(T_left_t)
    
    axes[1, 2].plot(t_maneuver, T_diff, 'purple', label='T_right - T_left', linewidth=2)
    axes[1, 2].plot(t_maneuver, T_sum, 'orange', label='T_right + T_left', linewidth=2)
    axes[1, 2].set_xlabel('Time [s]')
    axes[1, 2].set_ylabel('Thrust Combination [N]')
    axes[1, 2].set_title('Thrust Sum and Difference')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('thrust_allocation.png', dpi=150, bbox_inches='tight')
    print("  Saved: thrust_allocation.png")
    plt.close()


def plot_integrated_system():
    """Plot complete integrated system performance."""
    print("Generating integrated system plot...")
    
    # Create waypoint path
    waypoints = [
        [0, 0],
        [20, 0],
        [20, 20],
        [0, 20],
    ]
    path = WaypointPath(waypoints, method='cubic', closed=True)
    
    # Initialize controllers
    los = LOSGuidance()
    los.los_parameters(U=1.5, delta=5.0, k=0.5)
    
    mrac = MRAC(tether_length=3.5, epsilon=0.7, k_v=1.5, k_a=1.5)
    heading = HeadingController(k_psi=10.0, k_r=5.0, mode=HeadingMode.LOS)
    allocator = ThrustAllocator(thruster_separation=0.5, max_thrust=50.0)
    
    # Simulation
    dt = 0.1
    t_sim = 80.0
    steps = int(t_sim / dt)
    
    # Initial states
    towfish_pos = np.array([0.0, -5.0])  # Start off-track
    towfish_vel = np.array([0.0, 0.0])
    asv_pos = towfish_pos - np.array([3.5, 0.0])
    asv_vel = np.array([0.0, 0.0])
    asv_psi = 0.0
    asv_r = 0.0
    s = 0.0
    
    # Storage
    towfish_trajectory = []
    asv_trajectory = []
    cross_track_errors = []
    v_ref_prev = jnp.array([0.0, 0.0])
    
    for i in range(steps):
        # LOS guidance
        v_ref, s_dot = los.compute(
            position=towfish_pos.tolist(),
            s=s,
            path_function=path
        )
        
        # Update path parameter
        s += s_dot * dt
        s = max(0.0, min(s, path.get_path_length()))
        
        # v_ref_dot via finite difference
        v_ref_dot = (v_ref - v_ref_prev) / dt
        v_ref_prev = v_ref
        
        # MRAC control
        u_p, zeta_dot = mrac.compute(
            asv_position=asv_pos.tolist(),
            asv_velocity=asv_vel.tolist(),
            towfish_position=towfish_pos.tolist(),
            v_ref=v_ref,
            v_ref_dot=v_ref_dot,
            dt=dt
        )
        mrac.zeta += zeta_dot * dt
        
        # Heading control
        psi_ref, r_ref = heading.compute_reference_from_los(v_ref, v_ref_dot)
        tau_r = heading.compute(asv_psi, asv_r, psi_ref, r_ref)
        
        # Body frame surge force
        F_surge = u_p[0] * np.cos(asv_psi) + u_p[1] * np.sin(asv_psi)
        
        # Thrust allocation
        T_left, T_right = allocator.allocate(float(F_surge), float(tau_r))
        
        # Simple dynamics integration
        m_towfish = 25.0
        drag = 5.0
        towfish_acc = (np.array(u_p) - drag * towfish_vel) / m_towfish
        towfish_vel += towfish_acc * dt
        towfish_pos += towfish_vel * dt
        
        # ASV dynamics (simplified)
        m_asv = 50.0
        I_asv = 5.0
        F_total = T_left + T_right
        asv_acc_surge = (F_total - 10.0 * np.linalg.norm(asv_vel)) / m_asv
        
        # Update ASV velocity in body frame then convert to nav frame
        asv_vel_magnitude = np.linalg.norm(asv_vel) + asv_acc_surge * dt
        asv_vel = asv_vel_magnitude * np.array([np.cos(asv_psi), np.sin(asv_psi)])
        asv_pos += asv_vel * dt
        
        # Yaw dynamics
        asv_r += (tau_r / I_asv) * dt
        asv_psi += asv_r * dt
        asv_psi = (asv_psi + np.pi) % (2 * np.pi) - np.pi
        
        # Compute cross-track error
        path_point = path(s)
        cte = np.linalg.norm(towfish_pos - np.array(path_point))
        
        # Store
        towfish_trajectory.append(towfish_pos.copy())
        asv_trajectory.append(asv_pos.copy())
        cross_track_errors.append(cte)
    
    towfish_trajectory = np.array(towfish_trajectory)
    asv_trajectory = np.array(asv_trajectory)
    cross_track_errors = np.array(cross_track_errors)
    
    # Generate path points for plotting
    s_plot = np.linspace(0, path.get_path_length(), 200)
    path_points = np.array([path(s_i) for s_i in s_plot])
    
    # Create figure
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[2, 1])
    
    # Main trajectory plot
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(path_points[:, 0], path_points[:, 1], 'k--', 
             linewidth=2, label='Reference Path')
    ax1.plot(towfish_trajectory[:, 0], towfish_trajectory[:, 1], 
             'b-', linewidth=1.5, label='Towfish Trajectory', alpha=0.8)
    ax1.plot(asv_trajectory[:, 0], asv_trajectory[:, 1], 
             'r-', linewidth=1, label='ASV Trajectory', alpha=0.6)
    
    # Mark start and end
    ax1.plot(towfish_trajectory[0, 0], towfish_trajectory[0, 1], 
             'go', markersize=10, label='Start')
    ax1.plot(towfish_trajectory[-1, 0], towfish_trajectory[-1, 1], 
             'ro', markersize=10, label='End')
    
    # Add waypoint markers
    for wp in waypoints:
        ax1.plot(wp[0], wp[1], 'ks', markersize=8)
    
    ax1.set_xlabel('East [m]')
    ax1.set_ylabel('North [m]')
    ax1.set_title('Integrated System: Path Following Performance')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # Cross-track error
    ax2 = fig.add_subplot(gs[1, 0])
    time = np.linspace(0, t_sim, len(cross_track_errors))
    ax2.plot(time, cross_track_errors, 'b-', linewidth=2)
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Cross-Track Error [m]')
    ax2.set_title('Tracking Error')
    ax2.grid(True, alpha=0.3)
    
    # Statistics box
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis('off')
    
    stats_text = f"""
    Performance Statistics:
    
    Mean CTE: {np.mean(cross_track_errors):.3f} m
    Max CTE: {np.max(cross_track_errors):.3f} m
    Final CTE: {cross_track_errors[-1]:.3f} m
    
    Path Length: {path.get_path_length():.1f} m
    Distance Traveled: {np.sum(np.linalg.norm(np.diff(towfish_trajectory, axis=0), axis=1)):.1f} m
    
    Simulation Time: {t_sim:.1f} s
    """
    
    ax3.text(0.1, 0.5, stats_text, transform=ax3.transAxes,
             fontsize=12, verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
             family='monospace')
    
    plt.tight_layout()
    plt.savefig('integrated_system.png', dpi=150, bbox_inches='tight')
    print("  Saved: integrated_system.png")
    plt.close()


def main():
    """Generate all plots."""
    print("\n" + "="*60)
    print("Modular Controller Component Visualization")
    print("="*60 + "\n")
    
    plot_mrac_adaptation()
    plot_heading_controller_modes()
    plot_thrust_allocation()
    plot_integrated_system()
    
    print("\n" + "="*60)
    print("All plots generated successfully!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
