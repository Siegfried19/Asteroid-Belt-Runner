import time

import mujoco
import mujoco.viewer
import numpy as np

import manual_controller
import utility

MODEL_PATH = "environment.xml"
BODY_NAME = "spacecraft"
REPORT_PERIOD = 0.5
CAMERA_DISTANCE = 200.0
USE_MANUAL_CONTROL = True


def main():
    print("[main] Loading model...")
    model, data, body_id, joint_adr = _load_model()
    nu = model.nu
    print(f"[main] Model loaded: nu={nu}")

    use_manual = USE_MANUAL_CONTROL and nu > 0
    if use_manual:
        print("[main] Setting up manual controller...")
        manual_controller.setup(model)
        manual_controller.start_listener()

    next_report_time = time.time() + REPORT_PERIOD

    print("[main] Launching viewer...")
    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.cam.lookat[:] = data.body(body_id).xpos
            viewer.cam.distance = CAMERA_DISTANCE

            while viewer.is_running():
                step_start = time.time()
                viewer.cam.lookat[:] = data.body(body_id).xpos

                if use_manual:
                    if manual_controller.consume_flag('clear'):
                        _clear_controls(model, data, nu)
                    if manual_controller.consume_flag('reset'):
                        mujoco.mj_resetData(model, data)
                        _clear_controls(model, data, nu)
                        next_report_time = step_start
                    if manual_controller.consume_flag('quit'):
                        print("[main] Quit flag received, exiting loop.")
                        break

                    dt = float(model.opt.timestep)
                    delta = manual_controller.ctrl_delta(dt)
                    _apply_control_delta(model, data, nu, delta)

                mujoco.mj_step(model, data)
                viewer.sync()

                if step_start >= next_report_time:
                    utility.report_motion_status(model, data, joint_adr, nu)
                    next_report_time += REPORT_PERIOD

                time_until_next_step = model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
    finally:
        if use_manual:
            manual_controller.stop_listener()
        print("[main] Viewer closed.")


def _load_model(path=MODEL_PATH, body=BODY_NAME):
    model = mujoco.MjModel.from_xml_path(path)
    data = mujoco.MjData(model)
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{body}_joint")
    joint_adr = model.jnt_dofadr[joint_id]
    return model, data, body_id, joint_adr


def _clear_controls(model, data, nu):
    if nu == 0:
        return
    data.ctrl[:nu] = 0.0
    _clamp_ctrl(model, data, nu)


def _apply_control_delta(model, data, nu, delta):
    if nu == 0 or delta is None or delta.size == 0:
        return
    data.ctrl[:nu] += delta
    _clamp_ctrl(model, data, nu)


def _clamp_ctrl(model, data, nu):
    for i in range(nu):
        lo, hi = model.actuator_ctrlrange[i]
        if np.isfinite(lo) and data.ctrl[i] < lo:
            data.ctrl[i] = lo
        if np.isfinite(hi) and data.ctrl[i] > hi:
            data.ctrl[i] = hi


if __name__ == '__main__':
    main()
