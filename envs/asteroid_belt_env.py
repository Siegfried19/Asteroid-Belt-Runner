"""Gymnasium environment: fly the F8C through an asteroid belt (simplified dynamics).

Simplified dynamics = the single 6-DOF virtual thruster from `environment.xml`
(actuators Fx Fy Fz Mx My Mz). The agent commands normalized force/torque; the
goal is to traverse the belt along +X without colliding.

  observation : ship body-frame linear/angular velocity, orientation (fwd & up
                axes in world), goal direction + distance, and the K nearest
                asteroids (body-frame relative position, radius, surface distance).
  action      : Box(-1, 1, (6,)) -> linearly mapped to each actuator's ctrlrange.
  reward      : progress toward the goal plane, minus control & time cost,
                large bonus on success, large penalty on collision.
  episode end : collision / reached goal / out of bounds (terminated),
                or step budget exhausted (truncated).

The belt is regenerated with a fresh seed each reset when `randomize_belt=True`,
which rebuilds the MjModel — fine for headless training; the rollout viewer
re-acquires the model after reset.
"""

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from envs.belt_generator import BeltConfig, build_scene, SHIP_BODY


class AsteroidBeltEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        cfg: BeltConfig = None,
        dynamics: str = "simplified",
        k_nearest: int = 8,
        max_steps: int = 1500,
        frame_skip: int = 5,
        force_scale: float = 1.0,
        goal_clearance: float = 40.0,
        randomize_belt: bool = True,
        render_mode: str = None,
        # reward weights
        w_dist: float = 1.0,        # potential-based: reward for shrinking distance to goal point
        w_heading: float = 0.05,    # small bonus for pointing the nose toward the goal
        w_proximity: float = 0.4,   # penalty for hugging asteroid surfaces (teaches avoidance)
        d_safe: float = 12.0,       # proximity penalty kicks in within this surface distance (m)
        w_spin: float = 0.01,       # penalty on |angular velocity| (keep attitude controllable)
        ctrl_cost: float = 0.001,
        time_cost: float = 0.02,
        collision_penalty: float = 300.0,
        success_bonus: float = 200.0,
        oob_penalty: float = 100.0,
    ):
        super().__init__()
        self.base_cfg = cfg or BeltConfig()
        self.dynamics = dynamics
        self.k = k_nearest
        self.max_steps = max_steps
        self.frame_skip = frame_skip
        self.force_scale = force_scale
        self.goal_clearance = goal_clearance
        self.randomize_belt = randomize_belt
        self.render_mode = render_mode

        self.w_dist = w_dist
        self.w_heading = w_heading
        self.w_proximity = w_proximity
        self.d_safe = d_safe
        self.w_spin = w_spin
        self.ctrl_cost = ctrl_cost
        self.time_cost = time_cost
        self.collision_penalty = collision_penalty
        self.success_bonus = success_bonus
        self.oob_penalty = oob_penalty

        self.goal_x = self.base_cfg.belt_x_range[1] + goal_clearance
        self._reset_seed_counter = int(self.base_cfg.seed)

        self._build(self.base_cfg)

        # action: normalized vector -> the controlled actuators' ctrlrange
        self.action_space = spaces.Box(-1.0, 1.0, shape=(len(self.act_ids),), dtype=np.float32)
        obs_dim = 3 + 3 + 3 + 3 + 4 + self.k * 5
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)

        self._viewer = None
        self._steps = 0

    # ------------------------------------------------------------------ build
    def _build(self, cfg: BeltConfig):
        self.model, self.mj_spec, self.belt_info = build_scene(cfg, dynamics=self.dynamics)
        self.data = mujoco.MjData(self.model)
        self.body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, SHIP_BODY)
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{SHIP_BODY}_joint")
        self.qpos_adr = self.model.jnt_qposadr[jid]
        self.qvel_adr = self.model.jnt_dofadr[jid]
        self.ship_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "ship_collision"
        )
        # asteroid geom ids + radii (in belt_info order)
        self.ast_geom_ids = np.array(
            [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name) for name, _ in self.belt_info]
        )
        self.ast_radii = np.array([r for _, r in self.belt_info], dtype=float)
        # which actuators this env commands, and their ctrl ranges
        if self.dynamics == "realistic":
            from envs.thruster_layout import THRUSTER_NAMES
            self.act_ids = np.array(
                [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in THRUSTER_NAMES]
            )
        else:
            # the 6 virtual force/torque actuators (Fx Fy Fz Mx My Mz)
            self.act_ids = np.arange(6)
        self.ctrl_lo = self.model.actuator_ctrlrange[self.act_ids, 0].copy()
        self.ctrl_hi = self.model.actuator_ctrlrange[self.act_ids, 1].copy()

    # ------------------------------------------------------------------ helpers
    @property
    def ship_pos(self):
        return self.data.xpos[self.body_id]

    def _rot(self):
        return self.data.xmat[self.body_id].reshape(3, 3)

    def _asteroid_positions(self):
        if len(self.ast_geom_ids) == 0:
            return np.zeros((0, 3))
        return self.data.geom_xpos[self.ast_geom_ids]

    def _nearest_surface_dist(self):
        ast = self._asteroid_positions()
        if len(ast) == 0:
            return np.inf
        d = np.linalg.norm(ast - self.ship_pos, axis=1) - self.ast_radii
        return float(d.min())

    def _check_collision(self):
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if c.geom1 == self.ship_geom_id or c.geom2 == self.ship_geom_id:
                return True
        return False

    def _obs(self):
        R = self._rot()
        v_world = self.data.qvel[self.qvel_adr:self.qvel_adr + 3]
        w_local = self.data.qvel[self.qvel_adr + 3:self.qvel_adr + 6]
        v_body = R.T @ v_world
        fwd = R[:, 0]   # ship forward axis in world
        up = R[:, 2]

        pos = self.ship_pos
        goal_vec = np.array([self.goal_x, 0.0, 0.0]) - pos
        goal_dist = np.linalg.norm(goal_vec)
        goal_dir = goal_vec / (goal_dist + 1e-9)

        # K nearest asteroids
        ast = self._asteroid_positions()
        knn = np.zeros((self.k, 5), dtype=float)
        if len(ast) > 0:
            rel_world = ast - pos
            d = np.linalg.norm(rel_world, axis=1)
            order = np.argsort(d)[:self.k]
            for slot, idx in enumerate(order):
                rel_body = R.T @ rel_world[idx]
                surf = d[idx] - self.ast_radii[idx]
                knn[slot] = [rel_body[0], rel_body[1], rel_body[2], self.ast_radii[idx], surf]

        obs = np.concatenate([
            v_body, w_local, fwd, up,
            goal_dir, [goal_dist],
            knn.reshape(-1),
        ]).astype(np.float32)
        return obs

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self.randomize_belt:
            self._reset_seed_counter += 1
            cfg = BeltConfig(**{**self.base_cfg.__dict__, "seed": self._reset_seed_counter})
            self._build(cfg)
            if self._viewer is not None:
                self._sync_viewer_model()
        else:
            mujoco.mj_resetData(self.model, self.data)

        # ship starts at origin, identity orientation, at rest
        self.data.qpos[self.qpos_adr:self.qpos_adr + 7] = [0, 0, 0, 1, 0, 0, 0]
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self._steps = 0
        self._prev_goal_dist = self._goal_dist()
        return self._obs(), {}

    def _goal_dist(self):
        return float(np.linalg.norm(np.array([self.goal_x, 0.0, 0.0]) - self.ship_pos))

    def set_n_asteroids(self, n):
        """Curriculum hook: change belt density; takes effect on the next reset()."""
        self.base_cfg.n_asteroids = int(n)

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        ctrl = self.ctrl_lo + (action + 1.0) * 0.5 * (self.ctrl_hi - self.ctrl_lo)
        self.data.ctrl[:] = 0.0
        self.data.ctrl[self.act_ids] = ctrl * self.force_scale

        collided = False
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
            if self._check_collision():
                collided = True
                break

        self._steps += 1
        x = float(self.ship_pos[0])
        pos = self.ship_pos

        # potential-based shaping: reward shrinking the distance to the goal point
        # (flying off sideways increases the distance, so it's penalized automatically)
        goal_dist = self._goal_dist()
        reward = self.w_dist * (self._prev_goal_dist - goal_dist)
        self._prev_goal_dist = goal_dist
        # small heading bonus: nose (body +X) pointing toward the goal
        fwd = self._rot()[:, 0]
        goal_dir = (np.array([self.goal_x, 0.0, 0.0]) - pos)
        goal_dir = goal_dir / (np.linalg.norm(goal_dir) + 1e-9)
        reward += self.w_heading * float(np.dot(fwd, goal_dir))
        # proximity penalty: discourage hugging asteroid surfaces (learn to give them room)
        nearest_surf = self._nearest_surface_dist()
        if nearest_surf < self.d_safe:
            reward -= self.w_proximity * (self.d_safe - max(nearest_surf, 0.0))
        # spin penalty: keep attitude controllable so body-frame thrust stays useful
        w_local = self.data.qvel[self.qvel_adr + 3:self.qvel_adr + 6]
        reward -= self.w_spin * float(np.linalg.norm(w_local))
        reward -= self.ctrl_cost * float(np.sum(action ** 2))
        reward -= self.time_cost

        terminated = False
        truncated = False
        info = {}

        rho_yz = np.linalg.norm(pos[1:3])
        oob = (
            x < -50.0
            or rho_yz > self.base_cfg.belt_yz_radius + 60.0
            or x > self.goal_x + 30.0
        )

        if collided:
            reward -= self.collision_penalty
            terminated = True
            info["outcome"] = "collision"
        elif x >= self.goal_x:
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

        if self.render_mode == "human":
            self.render()
        return self._obs(), float(reward), terminated, truncated, info

    # ------------------------------------------------------------------ render
    def _sync_viewer_model(self):
        # passive viewer is bound to a specific model/data; rebuild it
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None

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
