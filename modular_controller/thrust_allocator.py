"""
Thrust allocator for differential thrust system.
Maps surge force and yaw torque to left/right thruster commands.
"""
import numpy as np


class ThrustAllocator:
    """
    Allocates surge force and yaw torque to differential thrusters.
    
    For a differential drive system with two thrusters separated by distance B:
    F_surge = T_left + T_right
    tau_r = (T_right - T_left) * B/2
    
    Solving for thruster commands:
    T_left = F_surge/2 - tau_r/B
    T_right = F_surge/2 + tau_r/B
    """
    
    def __init__(self, thruster_separation: float = 0.50, max_thrust: float = 50.0):
        """
        Initialize thrust allocator.
        
        Args:
            thruster_separation: Distance between left and right thrusters [m]
            max_thrust: Maximum thrust per thruster [N]
        
        Raises:
            ValueError: If parameters are invalid
        """
        if thruster_separation <= 0:
            raise ValueError("Thruster separation must be positive")
        if max_thrust <= 0:
            raise ValueError("Maximum thrust must be positive")
        
        self.B = thruster_separation
        self.max_thrust = max_thrust
    
    def allocate(self, F_surge: float, tau_r: float) -> tuple[float, float]:
        """
        Allocate surge force and yaw torque to left/right thrusters.
        
        Args:
            F_surge: Desired surge force in body frame [N]
            tau_r: Desired yaw torque [N⋅m]
            
        Returns:
            (T_left, T_right): Thruster commands [N]
        """
        # Compute raw thruster commands
        T_left = F_surge / 2.0 - tau_r / self.B
        T_right = F_surge / 2.0 + tau_r / self.B
        
        # Apply saturation limits
        T_left = np.clip(T_left, -self.max_thrust, self.max_thrust)
        T_right = np.clip(T_right, -self.max_thrust, self.max_thrust)
        
        return T_left, T_right
