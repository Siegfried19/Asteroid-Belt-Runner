import math
import threading

import numpy as np
from pynput import keyboard, mouse

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
    keyboard.KeyCode.from_char('u'): (3, -1),
    keyboard.KeyCode.from_char('o'): (3, +1),
    keyboard.KeyCode.from_char('j'): (4, -1),
    keyboard.KeyCode.from_char('l'): (4, +1),
    keyboard.KeyCode.from_char('i'): (5, -1),
    keyboard.KeyCode.from_char('k'): (5, +1),
}

POSITION_KEYS = {
    keyboard.KeyCode.from_char('d'): (0, +1),
    keyboard.KeyCode.from_char('a'): (0, -1),
    keyboard.KeyCode.from_char('w'): (1, +1),
    keyboard.KeyCode.from_char('s'): (1, -1),
    keyboard.KeyCode.from_char('v'): (2, +1),
    keyboard.KeyCode.from_char('c'): (2, -1),
}

SPECIAL_KEYS = {
    'clear': keyboard.Key.space,
    'reset': keyboard.Key.backspace,
    'quit': keyboard.Key.esc,
}

ctrl_lock = threading.Lock()
listener = None
mouse_listener = None

nu = 0
force_ctrl_speed_per_sec            = np.zeros(0, dtype=float)
position_speed_ctrl_speed_per_sec   = np.zeros(0, dtype=float)
angular_speed_ctrl_speed_per_sec    = np.zeros(0, dtype=float)

pressed_dirs    = np.zeros(0, dtype=float)
velocity_cmd    = np.zeros(6, dtype=float)
angular_speed   = None
position_speed  = None
flags = {'clear': False, 'reset': False, 'quit': False}
mouse_position = None


def setup(model, fraction=0.5, angular_deg_per_sec=180.0):
    global nu, ctrl_speed_per_sec, pressed_dirs, angular_speed, mouse_position

    nu = model.nu
    ctrl_speed_per_sec = np.zeros(nu, dtype=float)
    pressed_dirs = np.zeros(nu, dtype=float)
    angular_speed = math.radians(float(angular_deg_per_sec))
    mouse_position = None

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

def speed_control_setup(model, angular_degree_per_sec=180.0, position_speed_degree_per_sec=1080.0):
    global angular_speed, position_speed, pressed_dirs
    pressed_dirs = np.zeros(6, dtype=float)
    angular_speed = math.radians(float(angular_degree_per_sec))
    position_speed = 100
    with ctrl_lock:
        for name in flags:
            flags[name] = False
        velocity_cmd[:] = 0.0
    
    
def start_listener(suppress=True):
    global listener, mouse_listener
    listener = keyboard.Listener(on_press=_on_press_speed, on_release=_on_release_speed, suppress=suppress)
    listener.daemon = True
    listener.start()
    print("[manual_controller] Keyboard listener started.")
    # if mouse_listener is None:
    #     mouse_listener = mouse.Listener(on_move=_on_move)
    #     mouse_listener.daemon = True
    #     mouse_listener.start()


def stop_listener():
    global listener, mouse_listener
    if listener is not None:
        listener.stop()
        listener = None
    if mouse_listener is not None:
        mouse_listener.stop()
        mouse_listener = None


def control_update(dt):
    with ctrl_lock:
        ctrl_delta = pressed_dirs * ctrl_speed_per_sec * dt
    return ctrl_delta

def control_update_speed(dt):
    with ctrl_lock:
        speed_cmd = velocity_cmd.copy()
    return speed_cmd


def check_flag(name):
    with ctrl_lock:
        if not flags.get(name):
            return False
        flags[name] = False
        return True


def get_mouse_position():
    with ctrl_lock:
        return mouse_position


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
    # if key in ANGULAR_KEYS:
    #     axis, direction = ANGULAR_KEYS[key]
    #     with ctrl_lock:
    #         angular_velocity_cmd[axis] = direction * angular_speed


def _on_press_speed(key):
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
    # if key in KEY_TO_CTRL_INDEX:
    #     idx, direction = KEY_TO_CTRL_INDEX[key]
    #     if idx < nu:
    #         with ctrl_lock:
    #             pressed_dirs[idx] = direction
    if key in ANGULAR_KEYS:
        axis, direction = ANGULAR_KEYS[key]
        with ctrl_lock:
            velocity_cmd[axis] = direction * angular_speed
    if key in POSITION_KEYS:
        axis, direction = POSITION_KEYS[key]
        with ctrl_lock:
            velocity_cmd[axis] += direction * position_speed * 0.04


def _on_release(key):
    if key in KEY_TO_CTRL_INDEX:
        idx, _ = KEY_TO_CTRL_INDEX[key]
        if idx < nu:
            with ctrl_lock:
                pressed_dirs[idx] = 0.0
    # if key in ANGULAR_KEYS:
    #     axis, _ = ANGULAR_KEYS[key]
    #     with ctrl_lock:
    #         angular_velocity_cmd[axis] = 0.0
            
def _on_release_speed(key):
    if key in ANGULAR_KEYS:
        axis, _ = ANGULAR_KEYS[key]
        with ctrl_lock:
            velocity_cmd[axis] = 0.0

    # if key in POSITION_KEYS:
    #     axis, direction = POSITION_KEYS[key]
    #     with ctrl_lock:
    #         velocity_cmd[axis] -= direction * position_speed * 0.2


def _on_move(x, y):
    global mouse_position
    with ctrl_lock:
        mouse_position = (x, y)
