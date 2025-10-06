import numpy as np
import mujoco


def format_sim_time(t):
    ms = int(round((t - int(t)) * 1000))
    s = int(t) % 60
    m = (int(t) // 60) % 60
    h = int(t) // 3600
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def report_motion_status(model, data, joint_adr, nu, control = False):
    lin_vel = data.qvel[joint_adr: joint_adr + 3].copy()
    lin_acc = data.qacc[joint_adr: joint_adr + 3].copy()
    ang_vel = data.qvel[joint_adr + 3: joint_adr + 6].copy()
    ang_acc = data.qacc[joint_adr + 3: joint_adr + 6].copy()

    lin_speed = float(np.linalg.norm(lin_vel))
    lin_acc_mag = float(np.linalg.norm(lin_acc))
    ang_speed = float(np.linalg.norm(ang_vel))
    ang_acc_mag = float(np.linalg.norm(ang_acc))

    print(
        f"[t={format_sim_time(data.time)}] |v| = {lin_speed:.3f} m/s | "
        f"|a| = {lin_acc_mag:.3f} m/s^2 | |ω| = {ang_speed:.3f} rad/s | "
        f"|θ| = {ang_acc_mag:.3f} rad/s^2"
    )

    vx, vy, vz = lin_vel
    ax, ay, az = lin_acc
    wx, wy, wz = ang_vel
    alphax, alphay, alphaz = ang_acc

    print(
        " Linear v [m/s]:"
        f" vx={vx:.3f}, vy={vy:.3f}, vz={vz:.3f} | "
        "Angular  [rad/s]:"
        f" ωx={wx:.3f}, ωy={wy:.3f}, ωz={wz:.3f}"
    )
    print(
        " Linear a [m/s^2]:"
        f" ax={ax:.3f}, ay={ay:.3f}, az={az:.3f} | "
        "Angular  [rad/s^2]:"
        f" alphax={alphax:.3f}, alphy={alphay:.3f}, alphz={alphaz:.3f}"
    )

    if control:
        forces = data.actuator_force[:nu].copy()
        ctrls = data.ctrl[:nu].copy()
        lines = []
        for i in range(nu):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            lines.append(f"{name}: force={forces[i]:.3f}, ctrl={ctrls[i]:.3f}")
        if lines:
            print("Actuators[0..{}]: ".format(nu - 1) + " | ".join(lines))
