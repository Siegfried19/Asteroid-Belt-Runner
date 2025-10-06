import math
import numpy as np

_MAX_RATE_DEG = 120.0
_MAX_RATE_RAD = math.radians(_MAX_RATE_DEG)
_DEFAULT_LIMIT = np.array([_MAX_RATE_RAD, _MAX_RATE_RAD, _MAX_RATE_RAD], dtype=float)


class PIController:
    """Simple PI controller for three-axis angular velocity regulation."""

    def __init__(
        self,
        kp=None,
        ki=None,
        integral_limit=None,
    ):
        self.kp = np.array(kp if kp is not None else [1.2, 1.2, 1.2], dtype=float)
        self.ki = np.array(ki if ki is not None else [0.6, 0.6, 0.6], dtype=float)
        self.integral_limit = (
            np.array(integral_limit, dtype=float) if integral_limit is not None else _DEFAULT_LIMIT.copy()
        )
        self.integral = np.zeros(3, dtype=float)

    def reset(self):
        self.integral[:] = 0.0

    def update(self, error, dt):
        error = np.asarray(error, dtype=float)
        if error.shape != (3,):
            raise ValueError("error must be a length-3 array")
        if dt <= 0.0:
            return -self.kp * error - self.ki * self.integral

        self.integral += error * dt
        if self.integral_limit is not None:
            np.clip(self.integral, -self.integral_limit, self.integral_limit, out=self.integral)

        proportional = self.kp * error
        integral_term = self.ki * self.integral
        return -(proportional + integral_term)


class PIDController:
    """PID controller for three-axis angular velocity regulation."""

    def __init__(
        self,
        kp=None,
        ki=None,
        kd=None,
        integral_limit=None,
    ):
        self.kp = np.array(kp if kp is not None else [1.5, 1.5, 1.5], dtype=float)
        self.ki = np.array(ki if ki is not None else [0.6, 0.6, 0.6], dtype=float)
        self.kd = np.array(kd if kd is not None else [0.2, 0.2, 0.2], dtype=float)
        self.integral_limit = (
            np.array(integral_limit, dtype=float) if integral_limit is not None else _DEFAULT_LIMIT.copy()
        )
        self.integral = np.zeros(3, dtype=float)
        self.prev_error = np.zeros(3, dtype=float)
        self.first_update = True

    def reset(self):
        self.integral[:] = 0.0
        self.prev_error[:] = 0.0
        self.first_update = True

    def update(self, error, dt):
        error = np.asarray(error, dtype=float)
        if error.shape != (3,):
            raise ValueError("error must be a length-3 array")
        if dt <= 0.0:
            derivative = (error - self.prev_error)
            return -(
                self.kp * error + self.ki * self.integral + self.kd * derivative
            )

        self.integral += error * dt
        if self.integral_limit is not None:
            np.clip(self.integral, -self.integral_limit, self.integral_limit, out=self.integral)

        if self.first_update:
            derivative = np.zeros(3, dtype=float)
            self.first_update = False
        else:
            derivative = (error - self.prev_error) / dt

        self.prev_error = error.copy()

        proportional = self.kp * error
        integral_term = self.ki * self.integral
        derivative_term = self.kd * derivative
        return -(proportional + integral_term + derivative_term)
