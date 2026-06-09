"""Procedural asteroid-belt scene builder for the F8C Lightning sim.

Loads the base ship model (`environment.xml`) into a MuJoCo `MjSpec`, attaches a
collision proxy to the ship (the visual STL has collisions disabled), and scatters
a configurable belt of irregular "potato" asteroids across a slab along the +X axis.

Design (post-rebuild, 2026-06-08):
- Asteroids are **mesh geoms** drawn from the procedural library in `assets/asteroids/`
  (built by `envs/asteroid_mesh.py`). Each asteroid gets its own mesh asset so it can
  have a unique per-axis `scale` (= power-law base size x random aspect) and a random
  initial orientation, so no two rocks look alike. MuJoCo's collision uses the convex
  hull of the mesh (craters filled, bumps kept); the visible surface keeps its lumps.
- Each asteroid is a **free-joint body** (drifts + spins). The scene is compiled **once**;
  `AsteroidBeltEnv.reset()` re-places the rocks via qpos/qvel rather than recompiling.
- Power-law size distribution (many small rocks, few large), a minimum-separation check
  so rocks don't overlap, and a clear bubble around the ship spawn at the origin.
- `r_eff` (returned per asteroid) is a **conservative enclosing-sphere radius**
  (max-axis scale x the mesh's max vertex radius). Used by the env for the radar
  surface-distance and proximity reward; conservative => the radar warns slightly early.

Coordinate convention: the ship spawns at the origin and traverses toward +X. The
belt occupies x in `belt_x_range`, scattered within a `belt_yz_radius` cylinder
about the X axis. The far plane (x = belt_x_range[1] + clearance) is the goal.

Run `python Agent_tool/preview_belt.py` to eyeball a generated belt, or
`python envs/belt_generator.py` for a headless smoke test.
"""

import os
from dataclasses import dataclass, field

import mujoco
import numpy as np

SHIP_BODY = "spacecraft"
PARK_X = 5000.0   # x where unplaced asteroids are parked (far past the belt, out of play)

ASSET_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "assets", "asteroids")
MESHDIR_REL = os.path.join("assets", "asteroids")  # relative to environment.xml meshdir="."


@dataclass
class BeltConfig:
    n_asteroids: int = 40                  # fewer, bigger rocks, with wide gaps for the 26 m ship
    belt_x_range: tuple = (100.0, 500.0)   # slab the belt occupies along +X (ship punches through it)
    belt_yz_radius: float = 120.0          # asteroids fill this radius of the X axis (no skirt corridor)
    # power-law size distribution: many small rocks, few large ones
    size_min: float = 5.0                  # base radius lower bound (m) -- no tiny debris
    size_max: float = 16.0                 # base radius upper bound (m) -- big chunks
    size_power: float = 2.5                # pdf ~ s^-power; higher -> more small rocks
    aspect_range: tuple = (0.7, 1.35)      # per-axis stretch -> irregular (non-spherical) shapes
    min_gap: float = 55.0                  # min surface-to-surface clearance (m) -> wide margin for the
                                           #   ~26 m ship collision box (gaps ~2x the ship; learnable)
    spawn_clear_radius: float = 25.0       # keep asteroids this far from the ship spawn (origin)
    # dynamics: slow drift + spin (low relative velocity, structurally realistic)
    drift_speed: float = 1.5               # max |linear v| per rock (m/s), sampled at reset
    spin_speed: float = 0.4                # max |angular v| per rock (rad/s), sampled at reset
    density: float = 2500.0               # kg/m^3 (rocky) -> mass from convex-hull volume
    # ship collision proxy = oriented box hugging the STL (nose-tail X, wingspan Y, thin Z),
    # centred at +ship_box_cx along body-X. Collision is geometric (box vs asteroid sphere).
    ship_box_half: tuple = (12.21, 12.83, 2.93)
    ship_box_cx: float = 0.67
    n_mesh_library: int = 12               # how many base meshes exist in assets/asteroids/
    seed: int = 0
    asteroid_rgba: tuple = field(default_factory=lambda: (0.55, 0.5, 0.45, 1.0))


# Collision masks (kept for the ship proxy / realistic future). Ship-vs-asteroid contact
# is handled GEOMETRICALLY in the env (capsule vs conservative sphere), NOT by the MuJoCo
# contact solver: a high-speed ship ramming a light free-joint mesh asteroid produced
# energetic contacts that occasionally segfaulted the solver. Asteroids therefore carry
# contype/conaffinity = 0 (they drift + spin but generate no physical contacts).
SHIP_CONTYPE, SHIP_CONAFFINITY = 1, 2
AST_CONTYPE, AST_CONAFFINITY = 0, 0


@dataclass
class Asteroid:
    """Handle bundle for one placed asteroid (names resolved to ids by the env)."""
    body: str        # body name
    geom: str        # geom name
    joint: str       # free-joint name
    r_eff: float     # conservative enclosing-sphere radius (m)
    pos: tuple       # initial center position
    quat: tuple      # initial orientation (w, x, y, z)


def _mesh_max_radius(path: str) -> float:
    """Max vertex distance from origin for an OBJ mesh (unit-ish; ~1.0-1.6 here)."""
    rmax = 0.0
    with open(path) as f:
        for line in f:
            if line.startswith("v "):
                _, x, y, z = line.split()[:4]
                rmax = max(rmax, (float(x) ** 2 + float(y) ** 2 + float(z) ** 2) ** 0.5)
    return rmax


def _mesh_library(cfg: BeltConfig):
    """Return (paths, max_radii) for the procedural mesh library."""
    paths, radii = [], []
    for i in range(cfg.n_mesh_library):
        p = os.path.join(ASSET_DIR, f"asteroid_{i}.obj")
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"missing {p}; (re)build with `python envs/asteroid_mesh.py`")
        paths.append(p)
        radii.append(_mesh_max_radius(p))
    return paths, np.array(radii)


def _sample_sizes(n: int, cfg: BeltConfig, rng: np.random.Generator) -> np.ndarray:
    """Inverse-CDF sample of base radii with pdf ~ s^-power over [size_min, size_max]."""
    u = rng.uniform(size=n)
    a, b, p = cfg.size_min, cfg.size_max, cfg.size_power
    if abs(p - 1.0) < 1e-6:
        return a * (b / a) ** u
    lo, hi = a ** (1.0 - p), b ** (1.0 - p)
    return (lo + u * (hi - lo)) ** (1.0 / (1.0 - p))


def _rand_quat(rng: np.random.Generator):
    """Uniform random unit quaternion (Shoemake), returned as (w, x, y, z)."""
    u1, u2, u3 = rng.uniform(size=3)
    s1, s2 = np.sqrt(1.0 - u1), np.sqrt(u1)
    q1 = s1 * np.sin(2 * np.pi * u2)
    q2 = s1 * np.cos(2 * np.pi * u2)
    q3 = s2 * np.sin(2 * np.pi * u3)
    q4 = s2 * np.cos(2 * np.pi * u3)
    return (q4, q1, q2, q3)


def sample_belt(cfg: BeltConfig, rng: np.random.Generator, r_eff: np.ndarray):
    """Rejection-sample non-overlapping asteroid centers in the belt slab.

    Returns (positions Nx3, placed_index) — placed_index aligns surviving rocks back
    to their r_eff entry. Larger rocks are placed first (easier to fit the big ones).
    Asteroids that cannot be fit after the attempt budget are dropped (logged by caller).
    """
    order = np.argsort(-r_eff)  # big first
    placed_pos, placed_idx = [], []
    attempts_per = 200
    for idx in order:
        r = r_eff[idx]
        for _ in range(attempts_per):
            x = rng.uniform(*cfg.belt_x_range)
            rho = cfg.belt_yz_radius * np.sqrt(rng.uniform(0.0, 1.0))
            theta = rng.uniform(0.0, 2.0 * np.pi)
            y, z = rho * np.cos(theta), rho * np.sin(theta)
            p = np.array([x, y, z])
            if np.linalg.norm(p) < cfg.spawn_clear_radius + r:
                continue
            ok = True
            for q, j in zip(placed_pos, placed_idx):
                if np.linalg.norm(p - q) < r + r_eff[j] + cfg.min_gap:
                    ok = False
                    break
            if ok:
                placed_pos.append(p)
                placed_idx.append(idx)
                break
    return np.array(placed_pos).reshape(-1, 3), np.array(placed_idx, dtype=int)


def add_belt(spec: mujoco.MjSpec, cfg: BeltConfig):
    """Add free-joint mesh asteroids to `spec.worldbody`. Returns list[Asteroid].

    Each asteroid gets its own mesh asset (unique scale) referencing a random base
    mesh from the library. The scene is built once; the env re-places via qpos/qvel.
    """
    rng = np.random.default_rng(cfg.seed)
    paths, mesh_rmax = _mesh_library(cfg)

    n = cfg.n_asteroids
    lib_idx = rng.integers(0, cfg.n_mesh_library, size=n)
    sizes = _sample_sizes(n, cfg, rng)
    aspect = rng.uniform(cfg.aspect_range[0], cfg.aspect_range[1], size=(n, 3))
    scale = sizes[:, None] * aspect                       # per-axis mesh scale
    r_eff = sizes * aspect.max(axis=1) * mesh_rmax[lib_idx]  # conservative enclosing sphere

    # Always create exactly n asteroid bodies (fixed model size). Those that fit the belt
    # get a sampled position; any that don't fit start parked far out (the env re-places at
    # reset). In-belt asteroids are ordered FIRST so the curriculum's "first n_active" are all
    # in the belt.
    pos, placed = sample_belt(cfg, rng, r_eff)
    placed = [int(i) for i in placed]
    rest = [i for i in range(n) if i not in set(placed)]
    order = placed + rest                      # new index k -> original asteroid order[k]
    if rest:
        print(f"[belt_generator] {len(placed)}/{n} asteroids fit the belt "
              f"(min_gap={cfg.min_gap}); {len(rest)} start parked.")

    new_pos = np.zeros((n, 3))
    new_pos[:, 0] = PARK_X + 30.0 * np.arange(n)   # default parked
    new_pos[:len(placed)] = pos                    # in-belt come first, in sampled order
    lib_idx, scale, r_eff = lib_idx[order], scale[order], r_eff[order]

    wb = spec.worldbody
    asteroids = []
    for k in range(n):
        mesh_name = f"ast_mesh_{k}"
        m = spec.add_mesh()
        m.name = mesh_name
        m.file = os.path.join(MESHDIR_REL, f"asteroid_{int(lib_idx[k])}.obj")
        m.scale = [float(scale[k, 0]), float(scale[k, 1]), float(scale[k, 2])]

        quat = _rand_quat(rng)
        body = wb.add_body()
        body.name = f"asteroid_{k}"
        body.pos = [float(new_pos[k, 0]), float(new_pos[k, 1]), float(new_pos[k, 2])]
        body.quat = list(quat)
        jnt = body.add_freejoint()
        jnt.name = f"ast_joint_{k}"
        g = body.add_geom()
        g.name = f"ast_geom_{k}"
        g.type = mujoco.mjtGeom.mjGEOM_MESH
        g.meshname = mesh_name
        g.rgba = list(cfg.asteroid_rgba)
        g.density = cfg.density
        g.contype = AST_CONTYPE
        g.conaffinity = AST_CONAFFINITY
        asteroids.append(Asteroid(
            body=body.name, geom=g.name, joint=jnt.name, r_eff=float(r_eff[k]),
            pos=tuple(new_pos[k]), quat=quat))
    return asteroids


def add_ship_collision(spec: mujoco.MjSpec, cfg: BeltConfig):
    """Attach a collision-proxy box hugging the ship body (the STL geom is visual-only)."""
    ship = spec.body(SHIP_BODY)
    proxy = ship.add_geom()
    proxy.name = "ship_collision"
    proxy.type = mujoco.mjtGeom.mjGEOM_BOX
    proxy.size = list(cfg.ship_box_half)            # half-extents (nose-tail, wingspan, thickness)
    proxy.pos = [cfg.ship_box_cx, 0.0, 0.0]         # box centre offset along body-X
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
    """Build the full belt scene. Returns (model, spec, asteroids).

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
    asteroids = add_belt(spec, cfg)
    # translucent "exit zone" marker (visual only); the env moves it to the random goal each reset
    marker = spec.worldbody.add_geom()
    marker.name = "goal_marker"
    marker.type = mujoco.mjtGeom.mjGEOM_SPHERE
    marker.size = [25.0, 0, 0]
    marker.pos = [0.0, 0.0, 0.0]
    marker.contype = 0
    marker.conaffinity = 0
    marker.rgba = [0.2, 1.0, 0.3, 0.22]
    model = spec.compile()
    return model, spec, asteroids


if __name__ == "__main__":
    m, s, asts = build_scene()
    print(f"Built belt scene: nbody={m.nbody}, ngeom={m.ngeom}, "
          f"nmesh={m.nmesh}, asteroids={len(asts)}")
    if asts:
        r = np.array([a.r_eff for a in asts])
        print(f"r_eff: min={r.min():.2f} max={r.max():.2f} mean={r.mean():.2f}  "
              f"(power-law: {(r < r.mean()).mean()*100:.0f}% below mean)")
