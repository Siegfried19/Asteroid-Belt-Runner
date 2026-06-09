"""Gymnasium environment: fly the F8C through an asteroid belt (simplified dynamics).

Simplified dynamics = the single 6-DOF virtual thruster from `environment.xml`
(actuators Fx Fy Fz Mx My Mz). The agent commands normalized force/torque; the
goal is to traverse the belt along +X without colliding.

  observation : ship body-frame linear/angular velocity, orientation (fwd & up
                axes in world), goal direction + distance (16), plus a body-frame
                full-sphere radar: per direction bin, proximity (1/(1+surf)) and
                radial closing velocity of the nearest asteroid (2 x n_az x n_el).
  action      : Box(-1, 1, (6,)) -> F8C-calibrated force/torque (asymmetric main/
                retro thrust + RCS torques); realistic mode -> 17 thrusters.
  reward      : progress toward the goal plane, minus proximity / spin / G-load /
                control / time cost, large bonus on success, large penalty on collision.
  episode end : collision / reached goal / out of bounds (terminated),
                or step budget exhausted (truncated).

Scene model (post-rebuild, 2026-06-08): the belt is **compiled once** with
`n_asteroids` free-joint potato-mesh rocks. `reset()` re-places the rocks by writing
qpos (position + orientation) and qvel (slow drift + spin) directly — no recompile.
Curriculum density is handled by activating only `n_active` rocks and parking the
rest far beyond the bounds (so the model stays fixed-size while density varies).
"""

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from envs.belt_generator import BeltConfig, build_scene, sample_belt, _rand_quat, SHIP_BODY

# Parking spot for inactive (curriculum) asteroids: far past the goal, out of play.
_PARK_X = 5000.0


class AsteroidBeltEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        cfg: BeltConfig = None,
        dynamics: str = "simplified",
        k_nearest: int = 8,         # deprecated (KNN obs replaced by radar); kept for compat
        n_az: int = 12,             # radar azimuth bins (around body-X)
        n_el: int = 6,              # radar elevation bins
        max_steps: int = 1500,
        frame_skip: int = 5,
        force_scale: float = 1.0,
        oob_yz_margin: float = 25.0,  # lateral room beyond the belt edge before OOB (covers the
                                      #   ~13 m ship half-width + maneuvering; still no skirt corridor)
        # F8C-calibrated simplified-dynamics limits (decoupled from the XML envelope):
        fwd_accel: float = 103.5,   # main thrust +X (m/s^2, 10.55 G)
        rev_accel: float = 36.3,    # retro/brake -X (m/s^2, 3.70 G) -- asymmetric
        lat_accel: float = 40.0,    # lateral strafe Fy/Fz (m/s^2, no official figure)
        roll_rate: float = 140.0,   # max body rates (deg/s) used to size RCS torques
        pitch_rate: float = 38.0,
        yaw_rate: float = 35.0,
        rot_reach_time: float = 2.0,  # seconds to reach max rate -> sets angular accel
        goal_clearance: float = 40.0,
        # random exit: each episode the goal is a point (goal_x, gy, gz) at a random off-axis
        # spot, so the ship must weave to a specific exit instead of flying straight down +X.
        exit_r_min: float = 90.0,   # exit is sampled this far .. exit_r_max off the X axis (m)
        exit_r_max: float = 150.0,  #   (out toward the belt's 180 m rim -> forces real weaving)
        goal_radius: float = 60.0,  # DIAG(temp): big sphere -- verify PPO can learn to reach at all
        # --- goal mode --------------------------------------------------------------------------
        # "traverse"       : the goal is an (off-axis) exit point PAST the far plane -> fly through.
        # "interior_point" : the goal is a random, clearance-checked point INSIDE the belt -> fly in
        #                    to a specific spot and (optionally) stop there. obs/reward are unchanged
        #                    (the obs already carries body-frame goal dir/dist; reward is potential-
        #                    based to an arbitrary goal), so a traverse-trained net warm-starts here.
        goal_mode: str = "traverse",
        interior_min_depth: float = 200.0,  # interior goal X must be >= this far past the belt's near
                                            #   edge -> forces real penetration, no entrance-hugging goals
        interior_rho_frac: float = 0.7,     # interior goal lateral offset up to this fraction of belt rim
        interior_clearance: float = 30.0,   # min surface clearance (m) of the goal from every asteroid
        # arrival-speed tiers (interior_point): None -> reaching the sphere is success (tier 1). A float
        # -> must ALSO be slower than this at arrival (tier 2). A (lo, hi) via arrival_speed_random ->
        # per-episode random target, appended to the obs so the agent can comply (tier 3).
        arrival_speed: float = None,
        arrival_speed_random: tuple = None,
        randomize_belt: bool = True,
        render_mode: str = None,
        # reward weights
        w_dist: float = 1.5,        # potential progress: strong goal pull through the field (coll raised to match)
        w_speed: float = 0.0,       # OFF: speed reward keeps reviving the "ram" optimum; progress drives forward
        w_heading: float = 0.15,    # back to 0.15 (0.3 over-rotated; isolating the oob-fix effect)
        w_proximity: float = 6.0,   # avoidance: QUADRATIC penalty, but TIGHT so gaps stay passable
        w_closing: float = 0.8,     # avoidance MAIN LINE: strong closing-speed penalty (vs fear-based coll)
        d_safe: float = 15.0,       #   proximity radius -- < half a 55 m gap so threading isn't taxed
        d_close: float = 60.0,      #   closing-penalty radius -- WIDE early warning vs ramming a rock
        w_spin: float = 0.01,       # penalty on |angular velocity| (keep attitude controllable)
        w_gload: float = 0.15,      # penalty on linear G-load above g_safe (keep the pilot alive)
        g_safe: float = 8.0,        # comfortable sustained acceleration (G); excess is penalized
        ctrl_cost: float = 0.001,
        time_cost: float = 0.03,    # per-step cost -> rewards finishing fast
        collision_penalty: float = 600.0,   # lowered: 1200 made the ship FEAR forward motion -> retreat/stall
        success_bonus: float = 600.0,       # raised: make a clean traversal clearly worth the risk
        oob_penalty: float = 200.0,
        w_arrive: float = 0.0,      # OFF: arrival brake hurt (22%->10%); control precision is the real issue
        d_arrive: float = 120.0,    #   arrival-zone radius (m): start braking inside this goal distance
        w_retreat: float = 2.0,     # anti-retreat: penalty per metre behind the start line (x<0) -> no fleeing
    ):
        super().__init__()
        self.base_cfg = cfg or BeltConfig()
        self.dynamics = dynamics
        self.n_az = int(n_az)
        self.n_el = int(n_el)
        self.n_bins = self.n_az * self.n_el
        self.max_steps = max_steps
        self.frame_skip = frame_skip
        self.force_scale = force_scale
        self.oob_yz_margin = oob_yz_margin
        self.fwd_accel = fwd_accel
        self.rev_accel = rev_accel
        self.lat_accel = lat_accel
        self.roll_rate = roll_rate
        self.pitch_rate = pitch_rate
        self.yaw_rate = yaw_rate
        self.rot_reach_time = rot_reach_time
        self.goal_clearance = goal_clearance
        self.exit_r_min = exit_r_min
        self.exit_r_max = exit_r_max
        self.goal_radius = goal_radius
        self.goal_mode = goal_mode
        self.interior_min_depth = float(interior_min_depth)
        self.interior_rho_frac = float(interior_rho_frac)
        self.interior_clearance = float(interior_clearance)
        self.arrival_speed = arrival_speed
        self.arrival_speed_random = arrival_speed_random
        self._obs_arrival = arrival_speed_random is not None  # tier 3 appends the target to the obs
        self._arrival_target = arrival_speed                  # (re)sampled each reset
        self.randomize_belt = randomize_belt
        self.render_mode = render_mode

        self.w_dist = w_dist
        self.w_speed = w_speed
        self.w_heading = w_heading
        self.w_proximity = w_proximity
        self.w_closing = w_closing
        self.d_safe = d_safe
        self.d_close = d_close
        self.w_spin = w_spin
        self.w_gload = w_gload
        self.g_safe = g_safe
        self.ctrl_cost = ctrl_cost
        self.time_cost = time_cost
        self.collision_penalty = collision_penalty
        self.success_bonus = success_bonus
        self.oob_penalty = oob_penalty
        self.w_arrive = w_arrive
        self.d_arrive = d_arrive
        self.w_retreat = w_retreat

        self.goal_x = self.base_cfg.belt_x_range[1] + goal_clearance
        # fixed far plane for the OOB bound. In traverse mode the OOB far-bound rides goal_x (the
        # traverse curriculum tightens it as a "don't overshoot" signal); in interior mode the goal
        # sits inside the belt, so the OOB far-bound stays pinned here at the actual belt exit.
        self.belt_far_x = self.goal_x
        self.goal = np.array([self.goal_x, 0.0, 0.0])   # (re)sampled each reset

        self._build(self.base_cfg)
        self.n_active = self.n_total   # curriculum hook (set_n_asteroids); all active by default

        # action: normalized vector -> the controlled actuators' ctrlrange
        self.action_space = spaces.Box(-1.0, 1.0, shape=(len(self.act_ids),), dtype=np.float32)
        # obs = ego(16) + radar(2 channels x n_bins): proximity + closing velocity
        # (+1 for the per-episode arrival-speed target in interior tier-3 mode)
        obs_dim = 3 + 3 + 3 + 3 + 4 + 2 * self.n_bins + (1 if self._obs_arrival else 0)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)

        self._viewer = None
        self._steps = 0

    # ------------------------------------------------------------------ build
    def _build(self, cfg: BeltConfig):
        # Compile once; reset() re-places asteroids via qpos/qvel (no recompile).
        self.model, self.mj_spec, self.asteroids = build_scene(cfg, dynamics=self.dynamics)
        self.data = mujoco.MjData(self.model)
        self.body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, SHIP_BODY)
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{SHIP_BODY}_joint")
        self.qpos_adr = self.model.jnt_qposadr[jid]
        self.qvel_adr = self.model.jnt_dofadr[jid]
        self.ship_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "ship_collision"
        )
        self.goal_marker_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "goal_marker"
        )
        # ship collision proxy = oriented box hugging the STL — collisions are detected
        # geometrically against the asteroids' conservative spheres (r_eff), see _box_collision.
        self.ship_box_half = np.asarray(self.base_cfg.ship_box_half, dtype=float)
        self.ship_box_center = np.array([self.base_cfg.ship_box_cx, 0.0, 0.0])

        # per-asteroid handles (build order), and free-joint qpos/dof addresses
        self.n_total = len(self.asteroids)
        self.ast_geom_ids = np.array(
            [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, a.geom) for a in self.asteroids],
            dtype=int)
        self.ast_r = np.array([a.r_eff for a in self.asteroids], dtype=float)
        ast_jids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, a.joint)
                    for a in self.asteroids]
        self.ast_qpos_adr = np.array([self.model.jnt_qposadr[j] for j in ast_jids], dtype=int)
        self.ast_dof_adr = np.array([self.model.jnt_dofadr[j] for j in ast_jids], dtype=int)

        # Canonical belt layout from the build (in-belt asteroids first). reset() rotates this
        # about the X axis -> O(N) cheap reset, NO per-reset rejection sampling (which was both
        # slow and the source of the memory-corruption crashes).
        self._base_layout = np.array([a.pos for a in self.asteroids], dtype=float)

        # which actuators this env commands, and their ctrl ranges
        if self.dynamics == "realistic":
            from envs.thruster_layout import THRUSTER_NAMES
            self.act_ids = np.array(
                [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in THRUSTER_NAMES]
            )
            # realistic thrusters map action[-1,1] linearly onto their [0,max] ctrlrange
            self.ctrl_lo = self.model.actuator_ctrlrange[self.act_ids, 0].copy()
            self.ctrl_hi = self.model.actuator_ctrlrange[self.act_ids, 1].copy()
        else:
            # the 6 virtual force/torque actuators (Fx Fy Fz Mx My Mz). Map action[-1,1]
            # to F8C-calibrated force/torque via a per-axis, sign-asymmetric scale so that
            # action=0 -> zero force (no forward bias despite asymmetric main/retro thrust).
            self.act_ids = np.arange(6)
            mass = float(self.model.body_mass[self.body_id])
            # inertia about the body axes: rotate the principal moments back into the
            # body frame (model.body_inertia is in the principal frame given by body_iquat,
            # which may permute axes) and read the diagonal.
            Rq = np.zeros(9)
            mujoco.mju_quat2Mat(Rq, self.model.body_iquat[self.body_id])
            Rq = Rq.reshape(3, 3)
            I_body = Rq @ np.diag(self.model.body_inertia[self.body_id]) @ Rq.T
            Ix, Iy, Iz = I_body[0, 0], I_body[1, 1], I_body[2, 2]
            f_fwd = mass * self.fwd_accel
            f_rev = mass * self.rev_accel
            f_lat = mass * self.lat_accel
            t_roll = Ix * np.radians(self.roll_rate) / self.rot_reach_time
            t_pitch = Iy * np.radians(self.pitch_rate) / self.rot_reach_time
            t_yaw = Iz * np.radians(self.yaw_rate) / self.rot_reach_time
            # scale applied to positive vs negative action (Fx is asymmetric; rest symmetric)
            self.f_pos = np.array([f_fwd, f_lat, f_lat, t_roll, t_pitch, t_yaw])
            self.f_neg = np.array([f_rev, f_lat, f_lat, t_roll, t_pitch, t_yaw])

    # ------------------------------------------------------------------ helpers
    @property
    def ship_pos(self):
        return self.data.xpos[self.body_id]

    def _rot(self):
        return self.data.xmat[self.body_id].reshape(3, 3)

    def _active_slice(self):
        """Geom positions + radii of the currently active asteroids."""
        n = self.n_active
        if n == 0:
            return np.zeros((0, 3)), np.zeros(0)
        return self.data.geom_xpos[self.ast_geom_ids[:n]], self.ast_r[:n]

    def _active_vel(self):
        """World-frame linear velocities of the active asteroids (free-joint linear qvel)."""
        n = self.n_active
        if n == 0:
            return np.zeros((0, 3))
        return np.stack([self.data.qvel[d:d + 3] for d in self.ast_dof_adr[:n]])

    def _active_centers(self):
        """World positions + r_eff of the active asteroids, read straight from qpos. Valid even
        before mj_forward (each rock's free-joint qpos IS its body/geom position), so the interior
        goal sampler can clearance-check right after _place_asteroids writes the layout."""
        n = self.n_active
        if n == 0:
            return np.zeros((0, 3)), np.zeros(0)
        centers = np.stack([self.data.qpos[a:a + 3] for a in self.ast_qpos_adr[:n]])
        return centers, self.ast_r[:n]

    def _sample_interior_goal(self, rng):
        """A random point INSIDE the belt for goal_mode='interior_point': X at least
        `interior_min_depth` past the near edge (forces real penetration, no entrance-hugging),
        YZ uniform over a disc of `interior_rho_frac` of the belt rim, and surface-clear of every
        active asteroid by `interior_clearance` (the target never sits buried in a rock)."""
        cfg = self.base_cfg
        x_lo = cfg.belt_x_range[0] + self.interior_min_depth
        x_hi = cfg.belt_x_range[1]
        if x_lo >= x_hi:                       # min_depth too large for this belt -> clamp
            x_lo = max(cfg.belt_x_range[0], x_hi - 1.0)
        rho_cap = self.interior_rho_frac * cfg.belt_yz_radius
        centers, r = self._active_centers()
        g = None
        for _ in range(200):
            gx = rng.uniform(x_lo, x_hi)
            th = rng.uniform(0.0, 2.0 * np.pi)
            rho = rho_cap * np.sqrt(rng.uniform(0.0, 1.0))   # sqrt -> uniform over the disc area
            g = np.array([gx, rho * np.cos(th), rho * np.sin(th)])
            if len(centers) == 0:
                return g
            if float((np.linalg.norm(centers - g, axis=1) - r).min()) >= self.interior_clearance:
                return g
        return g   # rare: no clear point in 200 tries -> use the last candidate

    def _nearest_surface_dist(self):
        ast, r = self._active_slice()
        if len(ast) == 0:
            return np.inf
        d = np.linalg.norm(ast - self.ship_pos, axis=1) - r
        return float(d.min())

    def _nearest_obstacle(self):
        """Surface distance + closing speed to the single nearest active asteroid.
        closing > 0 means the gap is shrinking (ship heading straight into the rock)."""
        ast, r = self._active_slice()
        if len(ast) == 0:
            return np.inf, 0.0
        rel = ast - self.ship_pos
        d = np.maximum(np.linalg.norm(rel, axis=1), 1e-6)
        surf = d - r
        i = int(np.argmin(surf))
        ship_v = self.data.qvel[self.qvel_adr:self.qvel_adr + 3]
        rel_v = self._active_vel()[i] - ship_v
        closing = float(-np.dot(rel[i], rel_v) / d[i])   # +ve => approaching that rock
        return float(surf[i]), closing

    def _radar(self, R, pos, ship_v):
        """Body-frame full-sphere radar: per direction bin, the nearest obstacle.

        Two channels per bin: proximity = 1/(1+surface_dist) (closer -> larger), and
        closing velocity (radial, positive = approaching). No occlusion (analytic
        binning, near does not hide far); when two rocks share a bin, the nearer wins.
        Asteroids are treated as conservative enclosing spheres (r_eff).
        """
        prox = np.zeros(self.n_bins, dtype=np.float32)
        clos = np.zeros(self.n_bins, dtype=np.float32)
        ast, r = self._active_slice()
        if len(ast) == 0:
            return prox, clos

        rel = ast - pos                              # world-frame relative position
        d = np.linalg.norm(rel, axis=1)
        d = np.maximum(d, 1e-6)
        surf = d - r
        rel_body = rel @ R                           # = R.T @ rel per row (into body frame)
        az = np.arctan2(rel_body[:, 1], rel_body[:, 0])          # [-pi, pi]
        el = np.arcsin(np.clip(rel_body[:, 2] / d, -1.0, 1.0))   # [-pi/2, pi/2]
        az_bin = np.clip(((az + np.pi) / (2 * np.pi) * self.n_az).astype(int), 0, self.n_az - 1)
        el_bin = np.clip(((el + np.pi / 2) / np.pi * self.n_el).astype(int), 0, self.n_el - 1)
        flat = el_bin * self.n_az + az_bin

        rel_v = self._active_vel() - ship_v          # relative velocity (world)
        closing = -np.sum(rel * rel_v, axis=1) / d   # +ve => surface distance shrinking

        # fill each bin with its nearest (smallest surf) asteroid
        nearest = np.full(self.n_bins, np.inf)
        for i in range(len(ast)):
            b = flat[i]
            if surf[i] < nearest[b]:
                nearest[b] = surf[i]
                prox[b] = 1.0 / (1.0 + max(surf[i], 0.0))
                clos[b] = closing[i]
        return prox, clos

    def _box_collision(self):
        """Geometric collision: ship oriented box (hugging the STL) vs asteroid spheres.

        Ship-asteroid physical contacts are disabled in MuJoCo (they could segfault the
        solver at high closing speed), so 'crashing' = a rock's conservative sphere (r_eff)
        overlapping the ship's oriented bounding box (nose-tail x wingspan x thickness).
        """
        ast, r = self._active_slice()
        if len(ast) == 0:
            return False
        R = self._rot()
        p = (ast - self.ship_pos) @ R - self.ship_box_center   # asteroid centers in ship-box frame
        q = np.abs(p) - self.ship_box_half                     # per-axis distance outside the box
        d = np.linalg.norm(np.maximum(q, 0.0), axis=1)         # point-to-box distance (0 if inside)
        return bool((d - r <= 0.0).any())

    def _obs(self):
        R = self._rot()
        v_world = self.data.qvel[self.qvel_adr:self.qvel_adr + 3]
        w_local = self.data.qvel[self.qvel_adr + 3:self.qvel_adr + 6]
        v_body = R.T @ v_world
        fwd = R[:, 0]   # ship forward axis in world
        up = R[:, 2]

        pos = self.ship_pos
        goal_vec = self.goal - pos
        goal_dist = np.linalg.norm(goal_vec)
        goal_dir = goal_vec / (goal_dist + 1e-9)

        # full-sphere body-frame radar (proximity + closing velocity per direction bin)
        prox, clos = self._radar(R, pos, v_world)

        parts = [v_body, w_local, fwd, up, goal_dir, [goal_dist], prox, clos]
        if self._obs_arrival:   # tier 3: let the agent see the per-episode arrival-speed target
            parts.append([(self._arrival_target or 0.0) / 100.0])
        obs = np.concatenate(parts).astype(np.float32)
        return obs

    # ------------------------------------------------------------------ gym API
    def _place_asteroids(self, rng):
        """Cheap reset: take the build layout, rotate it about X, write qpos/qvel. O(N)."""
        # park everything first (far past the goal, at rest)
        for i in range(self.n_total):
            a, d = self.ast_qpos_adr[i], self.ast_dof_adr[i]
            self.data.qpos[a:a + 3] = [_PARK_X + 30.0 * i, 0.0, 0.0]
            self.data.qpos[a + 3:a + 7] = [1.0, 0.0, 0.0, 0.0]
            self.data.qvel[d:d + 6] = 0.0

        n = self.n_active
        if n == 0:
            return
        th = rng.uniform(0.0, 2.0 * np.pi)           # rotate the whole belt about X for variety
        ct, st = np.cos(th), np.sin(th)
        cfg = self.base_cfg
        for i in range(n):
            px, py, pz = self._base_layout[i]
            a, d = self.ast_qpos_adr[i], self.ast_dof_adr[i]
            self.data.qpos[a:a + 3] = [px, py * ct - pz * st, py * st + pz * ct]
            self.data.qpos[a + 3:a + 7] = np.asarray(_rand_quat(rng), dtype=float)
            dv = rng.normal(size=3); dv /= (np.linalg.norm(dv) + 1e-9)
            wv = rng.normal(size=3); wv /= (np.linalg.norm(wv) + 1e-9)
            self.data.qvel[d:d + 3] = dv * rng.uniform(0.0, cfg.drift_speed)
            self.data.qvel[d + 3:d + 6] = wv * rng.uniform(0.0, cfg.spin_speed)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # ship starts at origin, identity orientation, at rest
        self.data.qpos[self.qpos_adr:self.qpos_adr + 7] = [0, 0, 0, 1, 0, 0, 0]
        self.data.qvel[self.qvel_adr:self.qvel_adr + 6] = 0.0
        # scatter the asteroids for this episode FIRST: interior goals are clearance-checked
        # against them; traverse goals don't read them, so the order is harmless there.
        self._place_asteroids(self.np_random)
        # pick this episode's goal
        if self.goal_mode == "interior_point":
            self.goal = self._sample_interior_goal(self.np_random)
        else:
            # traverse: a random off-axis exit point past the far plane -> weave, don't fly straight
            th = self.np_random.uniform(0.0, 2.0 * np.pi)
            rho = self.np_random.uniform(self.exit_r_min, self.exit_r_max)
            self.goal = np.array([self.goal_x, rho * np.cos(th), rho * np.sin(th)])
        # arrival-speed target (interior tiers 2/3): None -> reaching the sphere alone is success
        if self.arrival_speed_random is not None:
            lo, hi = self.arrival_speed_random
            self._arrival_target = float(self.np_random.uniform(lo, hi))
        else:
            self._arrival_target = self.arrival_speed
        if self.goal_marker_id >= 0:                       # move the visible goal-zone marker
            self.model.geom_pos[self.goal_marker_id] = self.goal
            self.model.geom_size[self.goal_marker_id, 0] = self.goal_radius
        mujoco.mj_forward(self.model, self.data)
        self._steps = 0
        self._prev_goal_dist = self._goal_dist()
        return self._obs(), {}

    def _goal_dist(self):
        return float(np.linalg.norm(self.goal - self.ship_pos))

    def set_n_asteroids(self, n):
        """Curriculum hook: change active belt density; takes effect on the next reset()."""
        self.n_active = int(np.clip(n, 0, self.n_total))

    def set_exit_r(self, r_min, r_max):
        """Curriculum hook: change the exit off-axis distance range (m); next reset() picks it up.
        Ramping this up from near-0 lets the agent first learn to fly straight, then to weave."""
        self.exit_r_min = float(r_min)
        self.exit_r_max = float(max(r_max, r_min))

    def set_traverse(self, x_far):
        """Curriculum hook: set how far the ship must fly (goal X). Asteroids stay where they were
        built (full belt); only the goal distance ramps -- short traverse first, then longer."""
        self.goal_x = float(x_far) + self.goal_clearance

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        if self.dynamics == "realistic":
            ctrl = self.ctrl_lo + (action + 1.0) * 0.5 * (self.ctrl_hi - self.ctrl_lo)
        else:
            # sign-asymmetric scale: action=0 -> 0 force, +/-1 -> per-axis max in that direction
            scale = np.where(action >= 0.0, self.f_pos, self.f_neg)
            ctrl = action * scale
        self.data.ctrl[:] = 0.0
        self.data.ctrl[self.act_ids] = ctrl * self.force_scale

        collided = False
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
            if self._box_collision():
                collided = True
                break
        # safety: a high-speed contact can momentarily blow up the state; treat any
        # non-finite ship state as a collision rather than letting NaN reach the next step.
        if not np.isfinite(self.data.qpos[self.qpos_adr:self.qpos_adr + 7]).all() or \
           not np.isfinite(self.data.qvel[self.qvel_adr:self.qvel_adr + 6]).all():
            collided = True

        self._steps += 1
        x = float(self.ship_pos[0])
        pos = self.ship_pos

        # potential-based shaping: reward shrinking the distance to the goal point
        # (flying off sideways increases the distance, so it's penalized automatically)
        goal_dist = self._goal_dist()
        reward = self.w_dist * (self._prev_goal_dist - goal_dist)
        self._prev_goal_dist = goal_dist
        goal_dir = (self.goal - pos)
        goal_dir = goal_dir / (np.linalg.norm(goal_dir) + 1e-9)
        # speed reward: velocity component toward the goal (fly fast, in the right direction)
        v_world = self.data.qvel[self.qvel_adr:self.qvel_adr + 3]
        reward += self.w_speed * float(np.dot(v_world, goal_dir))
        # small heading bonus: nose (body +X) pointing toward the goal
        fwd = self._rot()[:, 0]
        reward += self.w_heading * float(np.dot(fwd, goal_dir))
        # avoidance: early + steep (quadratic) proximity penalty, plus a closing-speed penalty
        # that directly punishes barreling toward a nearby rock -- this is what breaks the
        # "ram straight through" optimum: heading fast at a near rock costs far more than the
        # progress it buys, while weaving around it (closing -> 0) is cheap.
        near_surf, near_closing = self._nearest_obstacle()
        # closing penalty: WIDE early warning -- punish barreling toward a rock. Weaving past it
        # (closing ~ 0, because the rock slides by laterally) is free, so this taxes "ram through"
        # without taxing gap-threading.
        if near_surf < self.d_close and near_closing > 0.0:
            reward -= self.w_closing * near_closing * (1.0 - near_surf / self.d_close)
        # proximity penalty: TIGHT + steep -- only when actually hugging a surface, so a ship
        # centred in a 55 m gap (~14 m clearance) is NOT penalised for passing through.
        if near_surf < self.d_safe:
            prox = (self.d_safe - max(near_surf, 0.0)) / self.d_safe   # 0..1, 1 at contact
            reward -= self.w_proximity * prox * prox                    # quadratic -> steep up close
        # spin penalty: keep attitude controllable so body-frame thrust stays useful
        w_local = self.data.qvel[self.qvel_adr + 3:self.qvel_adr + 6]
        reward -= self.w_spin * float(np.linalg.norm(w_local))
        # G-load penalty: discourage sustained high linear acceleration (would kill the pilot).
        # The ship *can* pull 10.55 G, but flying that hard is penalized above g_safe.
        a_lin = self.data.qacc[self.qvel_adr:self.qvel_adr + 3]
        g_load = float(np.linalg.norm(a_lin)) / 9.80665
        if g_load > self.g_safe:
            reward -= self.w_gload * (g_load - self.g_safe)
        reward -= self.ctrl_cost * float(np.sum(action ** 2))
        reward -= self.time_cost
        # arrival brake: once close to the goal, penalize speed so the ship slows to a stop INSIDE
        # the success sphere instead of blasting straight through it (overshoot was 100% of failures).
        if goal_dist < self.d_arrive:
            speed = float(np.linalg.norm(v_world))
            reward -= self.w_arrive * speed * (1.0 - goal_dist / self.d_arrive)
        # anti-retreat: the lazy escape is fleeing backward off the start (x<0) until x<-50 OOB.
        # Tax being behind the start line so retreating is never safer than pushing forward.
        if x < 0.0:
            reward -= self.w_retreat * (-x)

        terminated = False
        truncated = False
        info = {}

        rho_yz = np.linalg.norm(pos[1:3])
        # traverse: the far-bound rides goal_x (a tight "don't overshoot" brake signal that the
        # curriculum tightens); interior: the goal is inside, so the far-bound stays at the belt exit.
        far_bound = self.goal_x if self.goal_mode != "interior_point" else self.belt_far_x
        oob = (
            x < -50.0
            or rho_yz > self.base_cfg.belt_yz_radius + self.oob_yz_margin
            or x > far_bound + 30.0
        )

        if collided:
            reward -= self.collision_penalty
            terminated = True
            info["outcome"] = "collision"
        elif np.linalg.norm(pos - self.goal) < self.goal_radius and (
            self._arrival_target is None
            or float(np.linalg.norm(v_world)) <= self._arrival_target
        ):
            # reached the goal point (traverse: the random off-axis exit; interior: the in-belt
            # target). Tiers 2/3 additionally require the ship to be slow enough on arrival; if it's
            # in the sphere but too fast, this is False -> the episode continues so it can brake.
            reward += self.success_bonus
            terminated = True
            info["outcome"] = "success"
        elif oob:
            reward -= self.oob_penalty
            terminated = True
            info["outcome"] = "out_of_bounds"
        elif self._steps >= self.max_steps:
            truncated = True
            info["outcome"] = "timeout"

        if not np.isfinite(reward):
            reward = -self.collision_penalty
            terminated = True
            info["outcome"] = "collision"

        if self.render_mode == "human":
            self.render()
        obs = self._obs()
        if not np.isfinite(obs).all():
            obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        return obs, float(reward), terminated, truncated, info

    # ------------------------------------------------------------------ render
    def render(self):
        if self.render_mode == "human":
            import mujoco.viewer
            if self._viewer is None:
                self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
                self._viewer.cam.distance = 200.0
            self._viewer.cam.lookat[:] = self.ship_pos
            self._viewer.sync()
        elif self.render_mode == "rgb_array":
            renderer = mujoco.Renderer(self.model, 480, 640)
            renderer.update_scene(self.data)
            return renderer.render()

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
