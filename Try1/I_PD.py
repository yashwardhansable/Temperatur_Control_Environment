import numpy as np
import torch
import torch.nn as nn

"""
Input: Tensor
Output: Tensor
"""
class IPDController:
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp  # Proportional gain
        self.Ki = Ki  # Integral gain
        self.Kd = Kd  # Derivative gain

        self.last_error = 0.0
        self.integral = 0.0

    def update(self, error, dt=0.5):
        dt_tensor = torch.tensor([dt], dtype=torch.float32)

        # Integral term
        self.integral += error * dt_tensor
        integral = self.Ki * self.integral

        # Derivative term
        derivative = self.Kd * (error - self.last_error) / dt_tensor

        # Proportional term
        proportional = self.Kp * error

        output = proportional + integral + derivative

        # Update stored variables
        self.last_error = error

        return output

#
# Td = 222.37
# Ti = 889.84
#
# # Set up controller parameters
# Kp = 2.264
# Ki = Kp / Td
# Kd = Kp / Ti
#
# # Create an instance of the IPDController
# controller = IPDController(Kp, Ki, Kd)
#
# # Setpoint and initial measured value as tensors
# setpoint = torch.tensor([10.0], dtype=torch.float32)
# measured_value = torch.tensor([0.0], dtype=torch.float32)
#
# # Simulation parameters
# dt = 0.5  # Time step
# total_time = 1600  # Total simulation time
#
# # Simulation loop
# t = torch.arange(0.0, total_time, dt)
# for time in t:
#     output = controller.update(setpoint - measured_value, dt=dt)
#
#     # Update the process or system with the controller output
#     # In this example, we assume the measured value follows the setpoint
#     measured_value += output * dt
#
#     print("Time:", time.item(), "Setpoint:", setpoint.item(), "Measured:", measured_value.item(), "Output:",
#           output.item())
