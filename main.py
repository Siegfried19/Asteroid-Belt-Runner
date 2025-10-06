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
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, BODY_NAME)
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{BODY_NAME}_joint")
    joint_adr = model.jnt_dofadr[joint_id]
    nu = model.nu

    if USE_MANUAL_CONTROL:
        manual_controller.speed_control_setup(model)
        manual_controller.start_listener()

    next_report_time = time.time() + REPORT_PERIOD

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = data.body(body_id).xpos
        viewer.cam.distance = CAMERA_DISTANCE
        # cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "chase_onbody")
        # viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        # viewer.cam.fixedcamid = cam_id
        

        while viewer.is_running():
            step_start = time.time()
            viewer.cam.lookat[:] = data.body(body_id).xpos

            if USE_MANUAL_CONTROL:
                if manual_controller.check_flag('clear'):
                    _clear_controls(model, data, nu)
                if manual_controller.check_flag('reset'):
                    mujoco.mj_resetData(model, data)
                    _clear_controls(model, data, nu)
                    next_report_time = step_start
                if manual_controller.check_flag('quit'):
                    print("[main] Quit flag received, exiting loop.")
                    break

                dt = float(model.opt.timestep)
                update = manual_controller.control_update_speed(dt)
                # if isinstance(update, tuple):
                #     ctrl_delta, ang_vel_cmd = update
                # else:
                #     ctrl_delta, ang_vel_cmd = update, None
                # _apply_control(model, data, nu, ctrl_delta)
                # _apply_angular_velocity(data, joint_adr, ang_vel_cmd)
                _apply_velocity(data, joint_adr, update, body_id)

            mujoco.mj_step(model, data)
            viewer.sync()

            if step_start >= next_report_time:
                utility.report_motion_status(model, data, joint_adr, nu)
                next_report_time += REPORT_PERIOD

            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    if USE_MANUAL_CONTROL:
        manual_controller.stop_listener()
    print("viewer closed.")


def _clear_controls(model, data, nu):
    if nu == 0:
        return
    data.ctrl[:nu] = 0.0
    _clamp_ctrl(model, data, nu)


def _apply_control(model, data, nu, delta):
    if nu == 0 or delta is None or delta.size == 0:
        return
    data.ctrl[:nu] += delta
    _clamp_ctrl(model, data, nu)


def _apply_angular_velocity(data, joint_adr, ang_vel_cmd):
    if ang_vel_cmd is None:
        return
    if ang_vel_cmd.size != 3:
        return
    start = joint_adr + 3
    end = start + 3
    if end > data.qvel.size:
        return
    data.qvel[start:end] = ang_vel_cmd

def _apply_velocity(data, joint_adr, vel_cmd, body_id):
    R = data.xmat[body_id].reshape(3, 3)
    v_world = R @ vel_cmd[:3]
    omega_local = vel_cmd[3:6]
    data.qvel[joint_adr:joint_adr+3] = v_world
    data.qvel[joint_adr+3:joint_adr+6] = omega_local


def _clamp_ctrl(model, data, nu):
    for i in range(nu):
        lo, hi = model.actuator_ctrlrange[i]
        if np.isfinite(lo) and data.ctrl[i] < lo:
            data.ctrl[i] = lo
        if np.isfinite(hi) and data.ctrl[i] > hi:
            data.ctrl[i] = hi


if __name__ == '__main__':
    main()
