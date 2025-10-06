import math
import threading

import numpy as np
from pynput import keyboard

KEY_TO_CTRL_INDEX = {
    keyboard.KeyCode.from_char('d'): (0, +1),
    keyboard.KeyCode.from_char('a'): (0, -1),
    keyboard.KeyCode.from_char('w'): (1, +1),
    keyboard.KeyCode.from_char('s'): (1, -1),
    keyboard.KeyCode.from_char('q'): (2, +1),
    keyboard.KeyCode.from_char('e'): (2, -1),
    keyboard.KeyCode.from_char('1'): (3, -1),
    keyboard.KeyCode.from_char('2'): (3, +1),
    keyboard.KeyCode.from_char('3'): (4, +1),
    keyboard.KeyCode.from_char('4'): (4, -1),
    keyboard.KeyCode.from_char('5'): (5, -1),
    keyboard.KeyCode.from_char('6'): (5, +1),
}

ANGULAR_KEYS = {
    keyboard.KeyCode.from_char('7'): (0, -1),
    keyboard.KeyCode.from_char('8'): (0, +1),
    keyboard.KeyCode.from_char('9'): (1, +1),
    keyboard.KeyCode.from_char('0'): (1, -1),
    keyboard.KeyCode.from_char('-'): (2, -1),
    keyboard.KeyCode.from_char('='): (2, +1),
}

SPECIAL_KEYS = {
    'clear': keyboard.Key.space,
    'reset': keyboard.Key.backspace,
    'quit': keyboard.Key.esc,
}

ctrl_lock   = threading.Lock()
listener    = None
nu          = 0

ctrl_speed_per_sec      = np.zeros(0, dtype=float)
pressed_dirs            = np.zeros(0, dtype=float)
angular_velocity_cmd    = np.zeros(3, dtype=float)
angular_speed           = math.radians(10.0)

flags = {'clear': False, 'reset': False, 'quit': False}


def setup(model, fraction=0.5, angular_deg_per_sec=60.0):
    global nu, ctrl_speed_per_sec, pressed_dirs, angular_speed

    nu = model.nu
    ctrl_speed_per_sec = np.zeros(nu, dtype=float)
    pressed_dirs = np.zeros(nu, dtype=float)
    angular_speed = math.radians(float(angular_deg_per_sec))

    for i in range(nu):
        lo, hi = model.actuator_ctrlrange[i]
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            span = 1.0
        else:
            span = hi - lo
        ctrl_speed_per_sec[i] = fraction * span

    with ctrl_lock:
        for name in flags:
            flags[name] = False
        angular_velocity_cmd[:] = 0.0


def start_listener(suppress=True):
    global listener
    if listener is not None:
        return
    listener = keyboard.Listener(on_press=_on_press, on_release=_on_release, suppress=suppress)
    listener.daemon = True
    listener.start()


def stop_listener():
    global listener
    if listener is None:
        return
    listener.stop()
    listener = None


def control_update(dt):
    with ctrl_lock:
        ctrl_delta = pressed_dirs * ctrl_speed_per_sec * dt
        angular_cmd = angular_velocity_cmd.copy()
    return ctrl_delta, angular_cmd


def check_flag(name):
    with ctrl_lock:
        if not flags.get(name):
            return False
        flags[name] = False
        return True


def _on_press(key):
    if key == SPECIAL_KEYS['clear']:
        with ctrl_lock:
            flags['clear'] = True
        return
    if key == SPECIAL_KEYS['reset']:
        with ctrl_lock:
            flags['reset'] = True
        return
    if key == SPECIAL_KEYS['quit']:
        with ctrl_lock:
            flags['quit'] = True
        return
    if key in KEY_TO_CTRL_INDEX:
        idx, direction = KEY_TO_CTRL_INDEX[key]
        if idx < nu:
            with ctrl_lock:
                pressed_dirs[idx] = direction
    if key in ANGULAR_KEYS:
        axis, direction = ANGULAR_KEYS[key]
        with ctrl_lock:
            angular_velocity_cmd[axis] = direction * angular_speed


def _on_release(key):
    if key in KEY_TO_CTRL_INDEX:
        idx, _ = KEY_TO_CTRL_INDEX[key]
        if idx < nu:
            with ctrl_lock:
                pressed_dirs[idx] = 0.0
    if key in ANGULAR_KEYS:
        axis, _ = ANGULAR_KEYS[key]
        with ctrl_lock:
            angular_velocity_cmd[axis] = 0.0
