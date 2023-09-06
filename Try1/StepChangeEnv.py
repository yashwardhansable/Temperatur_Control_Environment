import numpy as np
class TemperatureWithDelayStepChange:
    def __init__(self, target_level, current_level):
        self.target_level = target_level  ##Target self.r
        self.current_level = current_level
        self.max_level = 10.0
        self.min_level = 0.0
        self.integral = 0.0
        self.prev_error = 0.0
        self.p = 0.0
        self.i = 0.0
        self.d = 0.0

        # Define the plant parameters
        self.K = 0.5  # Gain
        self.tau = 20  # Time constant
        self.theta = 5  # Time delay
        self.dt = 0.1  # Sampling time

        self.t_end = 1000  # Simulation time
        self.t = np.arange(0, self.t_end + self.dt, self.dt)  # Time vector
        self.r = np.ones(len(self.t))  # Reference signal (target temperature)
        one_fourth_length = len(self.t)//4
        three_fourth_length = 3*one_fourth_length
        half_length = len(self.t) // 2
        self.r[:half_length] = [self.target_level] * half_length
        self.r[half_length:] = [self.target_level + 10] * (len(self.t) - half_length)
        self.y = self.current_level * np.ones(len(self.t))  # Output signal (temperature)
        self.u = np.zeros(len(self.t))  # Control signal
        self.e = np.zeros(len(self.t))  # Error signal
        self.integral = 0.0  # Integral term
        self.s = 0

        self.current_step = 0

    def reset(self, target_level, current_level):
        self.current_level = current_level
        self.integral = 0.0
        self.prev_error = 0.0
        self.target_level = target_level
        done = False
        self.r = self.target_level * np.ones(len(self.t))  # Reference signal (target temperature)
        self.y = self.current_level * np.ones(len(self.t))  # Output signal (temperature)
        self.u = np.zeros(len(self.t))  # Control signal
        self.e = np.zeros(len(self.t))  # Error signal
        self.current_step = 0

        self.s = 0

        return self.target_level - self.current_level

    def step(self, action):
        p, i, d = action

        self.p = p
        self.i = i
        self.d = d
        e = self.r[self.current_step] - self.y[self.current_step - 1]
        # error = self.target_level - self.current_level

        # Calculate the proportional component
        proportional = e

        # Calculate the integral component
        self.integral += e * self.dt
        integral = self.integral

        # Calculate the derivative component
        derivative = (e - self.e[self.current_step - 1]) / self.dt
        # print(proportional, integral,derivative)

        # Update the control signal
        control_signal = self.p * proportional + self.i * integral + self.d * derivative

        self.u[self.current_step] = control_signal
        u_delayed = self.delay(self.u, int(self.theta / self.dt))
        self.s = u_delayed
        self.y[self.current_step] = (self.K * u_delayed[self.current_step - 1] * self.dt
                                     + self.tau * self.y[self.current_step - 1]) / (self.tau + self.dt)
        self.e[self.current_step] = e
        reward = -abs(e) * 1000
        self.current_step += 1
        done = False
        if (abs(e) < 0.001):
            reward += 10000
            done = True
        return self.e[self.current_step - 1], reward, done

    def delay(self, u, delay):
        """

        :param u: control signal
        :param delay: delay in signal
        :return: delayed signal
        """
        u_delayed = np.zeros(len(u))
        for i in range(len(u)):
            if i < delay:
                u_delayed[i] = 0
            else:
                u_delayed[i] = u[i - delay]
        return u_delayed
