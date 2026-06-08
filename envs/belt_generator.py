"""Procedural asteroid-belt scene builder for the F8C Lightning sim.

Loads the base ship model (`environment.xml`) into a MuJoCo `MjSpec`, attaches a
collision proxy to the ship (the visual STL has collisions disabled), scatters a
configurable belt of asteroid geoms across a slab along the +X axis, and compiles
to an `MjModel`. Building via MjSpec (rather than a giant hand-written XML) keeps
asteroid count / placement / seed fully parametric.

Coordinate convention: the ship spawns at the origin and traverses toward +X. The
belt occupies x in `belt_x_range`, scattered within a `belt_yz_radius` cylinder
about the X axis. The far plane (x = belt_x_range[1] + clearance) is the goal.

Run `python Agent_tool/preview_belt.py` to eyeball a generated belt.
"""

from dataclasses import dataclass, field

import mujoco
import numpy as np

SHIP_BODY = "spacecraft"


@dataclass
class BeltConfig:
    n_asteroids: int = 60
    belt_x_range: tuple = (80.0, 320.0)   # slab the belt occupies along +X
    belt_yz_radius: float = 60.0          # asteroids scattered within this radius of the X axis
    radius_range: tuple = (2.0, 10.0)     # asteroid sphere radii
    asteroid_density: float = 2500.0      # kg/m^3 (rocky); only matters if asteroids are dynamic
    dynamic: bool = False                 # False -> static geoms (cheap); True -> free-joint drifting rocks
    drift_speed: float = 0.0              # max |v| for dynamic asteroids (m/s), sampled per-rock at reset
    ship_collision_radius: float = 6.0    # collision-proxy capsule half-size (approx hull bound)
    ship_collision_halflen: float = 12.0
    spawn_clear_radius: float = 25.0      # keep asteroids this far from the ship spawn (origin)
    seed: int = 0
    asteroid_rgba: tuple = field(default_factory=lambda: (0.55, 0.5, 0.45, 1.0))


# Collision masks: ship and asteroids collide with each other, but asteroids do NOT
# collide among themselves (saves contacts) and nothing collides with the axis markers.
SHIP_CONTYPE, SHIP_CONAFFINITY = 1, 2
AST_CONTYPE, AST_CONAFFINITY = 2, 1


def _sample_positions(cfg: BeltConfig, rng: np.random.Generator):
    """Rejection-sample asteroid centers in the belt slab, clear of the spawn point."""
    xs, ys, zs, radii = [], [], [], []
    attempts = 0
    while len(xs) < cfg.n_asteroids and attempts < cfg.n_asteroids * 200:
        attempts += 1
        r = rng.uniform(*cfg.radius_range)
        x = rng.uniform(*cfg.belt_x_range)
        # uniform in a disk of radius belt_yz_radius (sqrt for area-uniformity)
        rho = cfg.belt_yz_radius * np.sqrt(rng.uniform(0.0, 1.0))
        theta = rng.uniform(0.0, 2.0 * np.pi)
        y, z = rho * np.cos(theta), rho * np.sin(theta)
        if np.linalg.norm([x, y, z]) < cfg.spawn_clear_radius + r:
            continue
        xs.append(x); ys.append(y); zs.append(z); radii.append(r)
    return np.array(xs), np.array(ys), np.array(zs), np.array(radii)


def add_belt(spec: mujoco.MjSpec, cfg: BeltConfig):
    """Add asteroid geoms to `spec.worldbody`. Returns list of (geom_or_body_name, radius)."""
    rng = np.random.default_rng(cfg.seed)
    xs, ys, zs, radii = _sample_positions(cfg, rng)
    wb = spec.worldbody
    info = []
    for i in range(len(xs)):
        pos = [float(xs[i]), float(ys[i]), float(zs[i])]
        r = float(radii[i])
        if cfg.dynamic:
            body = wb.add_body()
            body.name = f"asteroid_{i}"
            body.pos = pos
            g = body.add_geom()
            g.type = mujoco.mjtGeom.mjGEOM_SPHERE
            g.size = [r, 0, 0]
            g.rgba = list(cfg.asteroid_rgba)
            g.density = cfg.asteroid_density
            g.contype = AST_CONTYPE
            g.conaffinity = AST_CONAFFINITY
            body.add_freejoint()
            info.append((body.name, r))
        else:
            g = wb.add_geom()
            g.name = f"asteroid_{i}"
            g.type = mujoco.mjtGeom.mjGEOM_SPHERE
            g.size = [r, 0, 0]
            g.pos = pos
            g.rgba = list(cfg.asteroid_rgba)
            g.contype = AST_CONTYPE
            g.conaffinity = AST_CONAFFINITY
            info.append((g.name, r))
    return info


def add_ship_collision(spec: mujoco.MjSpec, cfg: BeltConfig):
    """Attach a collision-proxy capsule to the ship body (the STL geom is visual-only)."""
    ship = spec.body(SHIP_BODY)
    proxy = ship.add_geom()
    proxy.name = "ship_collision"
    proxy.type = mujoco.mjtGeom.mjGEOM_CAPSULE
    # capsule along the ship's body-X (forward) axis
    proxy.fromto = [-cfg.ship_collision_halflen, 0, 0, cfg.ship_collision_halflen, 0, 0]
    proxy.size = [cfg.ship_collision_radius, 0, 0]
    proxy.contype = SHIP_CONTYPE
    proxy.conaffinity = SHIP_CONAFFINITY
    proxy.rgba = [0.2, 0.8, 1.0, 0.0]  # invisible by default
    return proxy


def _disable_axis_collisions(spec: mujoco.MjSpec):
    """The world_axes marker capsules are visual only — keep them out of contact."""
    for g in spec.geoms:
        if g.name.startswith("axis_"):
            g.contype = 0
            g.conaffinity = 0


def build_scene(cfg: BeltConfig = None, base_xml: str = "environment.xml",
                dynamics: str = "simplified"):
    """Build the full belt scene. Returns (model, spec, belt_info).

    dynamics="simplified" -> use the 6 virtual force/torque actuators from the XML.
    dynamics="realistic"  -> additionally add the 17 physical thrusters (the env then
                             commands only those; the 6 virtual ones stay at ctrl=0).
    """
    cfg = cfg or BeltConfig()
    spec = mujoco.MjSpec.from_file(base_xml)
    _disable_axis_collisions(spec)
    add_ship_collision(spec, cfg)
    if dynamics == "realistic":
        from envs.thruster_layout import add_thrusters
        add_thrusters(spec, SHIP_BODY)
    belt_info = add_belt(spec, cfg)
    model = spec.compile()
    return model, spec, belt_info


if __name__ == "__main__":
    m, s, info = build_scene()
    print(f"Built belt scene: nbody={m.nbody}, ngeom={m.ngeom}, asteroids={len(info)}")
