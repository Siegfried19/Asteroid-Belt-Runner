"""Shared helper: draw a flame plume at each F8C thruster nozzle, sized by thrust.

Used by `thruster_flame_viewer.py` (live, thrust-driven) and `thruster_layout_viewer.py`
(ship pinned, static nozzle map). A flame is a capsule that shoots OUT of the nozzle in the
exhaust direction (opposite the thrust force), with length/width/color scaled by the
per-thruster intensity (thrust / max_thrust, in [0, 1]).

Render-only: the flames are appended to the viewer's `user_scn` as decorative geoms; they
never touch physics. Build a realistic scene first (`build_scene(..., dynamics="realistic")`)
so the `{name}_site` nozzle sites exist.
"""
import mujoco
import numpy as np

from envs.thruster_layout import THRUSTERS, THRUSTER_NAMES

# group -> base RGB (used when coloring by group, e.g. the static layout map)
GROUP_RGB = {
    "main":    (1.00, 0.45, 0.05),   # orange
    "reverse": (1.00, 0.15, 0.10),   # red
    "rcs":     (0.25, 0.65, 1.00),   # cyan/blue
}


def _heat_rgba(f):
    """Flame heat color: deep orange (low thrust) -> near-white yellow (full thrust)."""
    f = float(np.clip(f, 0.0, 1.0))
    return np.array([1.0, 0.30 + 0.60 * f, 0.05 + 0.30 * f, 0.85], dtype=np.float32)


def site_world(model, data, name):
    """World position + 3x3 rotation of a thruster's nozzle site."""
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{name}_site")
    return data.site_xpos[sid].copy(), data.site_xmat[sid].reshape(3, 3).copy()


def add_flame_geoms(user_scn, model, data, intensity, color_by_group=False,
                    base_len=1.5, scale_len=7.5, min_f=1e-3):
    """Append one flame capsule per active thruster to `user_scn`.

    intensity: array len(THRUSTERS) of thrust/max in [0,1].
    color_by_group: True -> color by thruster group (identify nozzles); False -> heat color.
    """
    for i, (name, _pos, direction, _thrust, group) in enumerate(THRUSTERS):
        f = float(intensity[i])
        if f <= min_f:
            continue
        if user_scn.ngeom >= user_scn.maxgeom:
            break
        p, R = site_world(model, data, name)
        d = np.asarray(direction, dtype=float)
        d = d / (np.linalg.norm(d) + 1e-12)
        exhaust = -(R @ d)                       # plume shoots opposite the thrust force
        tip = p + exhaust * (base_len + scale_len * f)

        if color_by_group:
            r, g, b = GROUP_RGB.get(group, (1.0, 1.0, 1.0))
            rgba = np.array([r, g, b, 0.85], dtype=np.float32)
        else:
            rgba = _heat_rgba(f)

        geom = user_scn.geoms[user_scn.ngeom]
        mujoco.mjv_initGeom(
            geom, mujoco.mjtGeom.mjGEOM_CAPSULE,
            np.zeros(3), np.zeros(3), np.zeros(9), rgba)
        width = 0.35 + 0.9 * f
        mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, width, p, tip)
        geom.label = name.encode("utf-8") if color_by_group else b""
        user_scn.ngeom += 1


def intensities_from(active: dict):
    """Build a full intensity vector from a {thruster_name: intensity} dict."""
    arr = np.zeros(len(THRUSTER_NAMES))
    for i, n in enumerate(THRUSTER_NAMES):
        if n in active:
            arr[i] = float(active[n])
    return arr
