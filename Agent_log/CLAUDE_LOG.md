# Claude Code 操作日志

记录 Claude Code 在本仓库的实质性操作，便于跨会话 / 跨平台接续。
规约见 `README.md`（每次操作后追加，不记录用户提问）。

## 2026-06-08 — 项目脚手架 + 总体计划落地

- **确认运行环境**：本仓库 RL 用 conda env `space-robotics-project`（Python 3.10），
  已装 MuJoCo 3.3.7 / Gymnasium 0.29.1 / Stable-Baselines3 2.3.0 / PyTorch 2.2.1+cu121 / pynput。
  缺 trimesh（仅 `STL_inertia_calculator.py` 用）。注意此 env numpy=1.26，与根目录
  `requirements.txt` 写的 2.3.3 不一致 —— RL 以本 env 为准。
- **建立目录约定（借鉴 Human_exo_interaction）**：新建 `Agent_tool/`（一次性小工具）、
  `Agent_log/`（本日志，git 跟踪并 push）、`TODO_list/`（计划）、`logs/`（训练输出，git 忽略）。
- **改 `.gitignore`**：忽略 `logs/ runs/ checkpoints/ MUJOCO_LOG.TXT *.zip wandb/` 等训练产物，
  但 `Agent_log/` 保持跟踪。
- **写总体计划** `TODO_list/PROJECT_PLAN.md`：6 阶段路线图（脚手架→小行星带→简化RL环境→
  训练→真实17推进器动力学→真实动力学训练），含开放问题的默认裁定。
- 现有 `main.py` 的直接速度控制（写 `data.qvel`）定位为"试玩"入口，不参与 RL；
  RL 动作用 XML 里的 6 个力/力矩执行器（简化动力学）。

## 2026-06-08 — Phase 1+2+3 落地（小行星带 / RL 环境 / 训练管线）

- **Phase 1** `envs/belt_generator.py`：用 `mujoco.MjSpec` 运行时组装场景。从 `environment.xml`
  加载 → 给飞船加隐形胶囊**碰撞代理**（STL 几何仍只做可视，原本 contype/conaffinity=0）→
  在 +X 轴向 slab 内 rejection-sample 散布 N 颗球形小行星（远离原点出生区）。碰撞掩码：
  飞船(contype1/aff2) vs 小行星(contype2/aff1) 相撞，小行星互不撞、且禁用坐标轴几何碰撞。
  支持 `dynamic=True` 漂移小行星（freejoint）。冒烟：nbody=3 ngeom=65（60 行星）。
- **Phase 2** `envs/asteroid_belt_env.py`：`AsteroidBeltEnv(gym.Env)`。obs 56 维
  (v_body3+ω3+fwd3+up3+goal_dir3+goal_dist1+最近K=8×(rel3+r1+surf1))，action 6 维映射到
  执行器 ctrlrange。reward=+X 进度 −控制 −时间，碰撞−100/到达+200/越界−50 终止，超时截断。
  **踩坑**：原把 MjSpec 存成 `self.spec`，与 Gymnasium 保留属性 `env.spec`(EnvSpec) 冲突，
  env_checker 报 `'MjSpec' has no attribute 'nondeterministic'` → 改名 `self.mj_spec`。
  `Agent_tool/check_env.py`：env_checker 通过 + 20 集随机 rollout（timeout8/oob10/collision2，
  无意外成功，符合预期）。
- **Phase 3** `train/train_ppo.py`：SB3 PPO + SubprocVecEnv(8) + CheckpointCallback + EvalCallback
  + tensorboard，输出到 `logs/<run>/`。`Agent_tool/rollout_viewer.py`（randomize_belt=False
  保持 viewer 句柄有效）。**冒烟训练通过**：~5000 fps，2 次 rollout + PPO 更新正常，
  ep_rew −78→−66（仅噪声，证明管线通），模型保存后清理 `logs/smoke`。
- 更新 `CLAUDE.md`（新增 Setup/Run、工作约定、RL stack 段）与 `TODO_list/PROJECT_PLAN.md`
  （勾掉 Phase 0/1/2，Phase 3 剩"正式训练"一项）。
- **下一步**：跑一次正式训练（1M~3M 步）看 reward 是否上升、能否学会穿越；之后进 Phase 4
  真实 17 推进器动力学。尚未启动正式训练（等用户确认是否现在长跑）。

## 2026-06-08 — 正式训练 v1 + Phase 4 真实动力学 + reward 翻车与修复

- **训练 v1**（`logs/ppo_simplified_v1`，2M 步，8 envs，~5000fps）：eval 回报从 −3 升到 100~135，
  曲线明显学习。**但用 `Agent_tool/eval_policy.py` 跑 100 集发现：mean_return 95，success 仅 4%，
  collision 17%，out_of_bounds 79%**。诊断：x-progress reward 可被"猛冲 +X 刷分后侧向飞出界"刷满，
  策略根本不需要到达目标 → reward 设计缺陷，非代码 bug。（v1 checkpoint 保留未删，供对照。）
- **修复 reward（env v2）**：改 potential-based —— `reward = w_dist*(prev_goal_dist - goal_dist)`
  （到目标点 (goal_x,0,0) 的距离减小量；侧飞增大距离→自动惩罚）+ `w_heading*dot(nose,goal_dir)`
  朝向小奖励；oob_penalty 50→100，ctrl_cost 0.01→0.001。参数名 w_progress/w_goal_dist 删除。
- **Phase 4 真实动力学建模**：新建 `envs/thruster_layout.py` —— 17 推进器（2 主 +X 5e6 / 3 反向 −X
  1.6e6 / 12 RCS 1e6），每个 = ship 上的 site + site-actuator(gear=推力方向 单向 ctrlrange[0,max])。
  off-COM site 自然产生力矩。**wrench 矩阵 rank=6、每轴 ± 均可达，6-DOF 可控**（`python envs/thruster_layout.py` 验证）。
  `build_scene(dynamics=...)` + env `dynamics` 开关切换简化(6 actions)/真实(17 actions)；真实模式
  只命令 17 推进器执行器（6 虚拟留 0，因 MjSpec 3.3 不支持删 actuator）。`Agent_tool/check_env.py`
  双模式均过 env_checker。
- **新工具**：`Agent_tool/eval_policy.py`（成功率/结局分布）、`rollout_viewer.py`、`preview_belt.py`。
  train_ppo.py 加 `--dynamics` 开关。
- **启动训练 v2**（`logs/ppo_simplified_v2`，2M 步，potential reward）后台跑。待评估成功率。
- 注：v1 的 stdout 因我误删了重定向目标文件而丢失（进程仍写已删 fd），但 checkpoint/monitor/eval 齐全；
  v2 起改重定向到 `logs/train_v2.out`（已 gitignore）。

## 2026-06-08 — v2 评估 + reward 再迭代（避障）+ 课程学习 → v3

- **v2 评估**（potential reward，100 集）：success 7%、**collision 72%**、oob 21%（v1 是 oob 79%）。
  结论：potential reward **修好了"侧飞出界"**（oob 79%→21%，飞船现在朝目标走对了），**但变成
  直冲撞石头**——除了终端 −100 没有提前避障激励，而直冲能先攒 ~360 potential，撞了也"划算"。
- **reward 再迭代（env v3 参数）**：
  - 加 **proximity 惩罚** `w_proximity=0.4, d_safe=12`：离最近小行星表面 <12m 就按 (12−surf) 线性扣分，
    教它提前给石头让路。
  - 加 **spin 惩罚** `w_spin=0.01`：罚 |角速度|，保持姿态可控（body-frame 推力才有用）。
  - collision_penalty 100→**300**（撞了不再划算）。
- **课程学习**（Phase 7 提前做）：env 加 `set_n_asteroids()`，train 加 `CurriculumCallback`
  （前 60% 训练把密度从 n_start 线性升到 n_asteroids）+ `--curriculum/--n-start` 开关。
  通过 `env_method` 下发到各 SubprocVecEnv worker，下个 reset 生效。
- **启动 v3**（`logs/ppo_simplified_v3`，3M 步，curriculum 12→60，避障+spin reward）后台跑。待评估。
- v1/v2 checkpoint 全部保留未删，供对照。

## 2026-06-08 — 用户叫停、地基重建（决策已定）

- **现状诚实评估呈给用户**：管线齐全但"控制器没学会"(success 4→7%，v3 发散到 −221 已 kill)，
  且地基是占位的：小行星带是个圆柱走廊塞随机球(不像真实带)、尺度/单位拍脑袋、简化动力学力大到瞬间机动。
  用户认同"很多东西要重建"，叫停训练迭代，先重建地基。
- **用户三项决策**：①尺度=**游戏化标度**(紧凑、只让结构真实)；②RL 动力学=**简化6力先行**
  (但我要把力调到合理量级、非瞬间机动，之后再迁 17 推进器)；③小行星=**缓慢漂移+自转**(动态)。
- **小行星带按"结构真实、密度人为调高"重建**(用户授权我自行设计、不用先看)：幂律尺寸分布(小石头远多于大)、
  椭球不规则外形+随机朝向、低相对速度漂移+自转、最小间距防重叠。完全照真实密度会空无一物→无意义，故调高密度。
- **重建范围**：①重写 `belt_generator.py`(幂律/椭球/动态漂移自转/间距)；②env 改：动态小行星(free-joint,
  reset 时下发漂移/自转 qvel)、简化力调到合理上限(max_force/max_torque 解耦 XML 巨值)、obs 增加小行星相对速度；
  ③更新工具；④重训。v1/v2 旧 run 保留但已过时。
