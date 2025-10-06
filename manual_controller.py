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

SPECIAL_KEYS = {
    'clear': keyboard.Key.space,
    'reset': keyboard.Key.backspace,
    'quit': keyboard.Key.esc,
}

ctrl_lock   = threading.Lock()
listener    = None
nu          = None
ctrl_speed_per_sec  = None
pressed_dirs        = None
flags = {'clear': False, 'reset': False, 'quit': False}


def setup(model, fraction=0.5):
    global nu, ctrl_speed_per_sec, pressed_dirs
    
    nu = model.nu
    ctrl_speed_per_sec = np.zeros(nu, dtype=float)
    pressed_dirs = np.zeros(nu, dtype=float)

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


def start_listener(suppress=True):
    global listener
    listener = keyboard.Listener(on_press=_on_press, on_release=_on_release, suppress=suppress)
    listener.daemon = True
    listener.start()


def stop_listener():
    listener.stop()
    listener = None


def output_update(dt):
    with ctrl_lock:
        return pressed_dirs * ctrl_speed_per_sec * dt


def reset_flag(name):
    with ctrl_lock:
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


def _on_release(key):
    if key in KEY_TO_CTRL_INDEX:
        idx, _ = KEY_TO_CTRL_INDEX[key]
        with ctrl_lock:
            pressed_dirs[idx] = 0.0
