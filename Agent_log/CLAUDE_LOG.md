# Claude Code 操作日志

记录 Claude Code 在本仓库的实质性操作，便于跨会话 / 跨平台接续。
规约见 `README.md`（每次操作后追加，不记录用户提问）。

## 2026-06-08 — 项目脚手架 + 总体计划落地

- **确认运行环境**：本仓库 RL 用 conda env `asteroid-belt-runner`（Python 3.10），
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

## 2026-06-08 — 换机器：重建 conda 环境

- **新机器**：原 `asteroid-belt-runner` env 不存在（只有 base / myoassist），GPU = RTX 4090 (24G)。
- **重建 env**：`conda create -n asteroid-belt-runner python=3.10` + pip 装
  mujoco==3.3.7 / gymnasium==0.29.1 / stable-baselines3==2.3.0 / numpy<2(→1.26.4) /
  torch==2.2.1(cu121) / pynput / tensorboard。版本与旧机器一致。
- **验证**：`torch.cuda.is_available()=True`(4090)；`Agent_tool/check_env.py` 双模式均
  env_checker 通过 + 20 集随机 rollout 正常。新机器可跑。
- 注：仍未补 `requirements-rl.txt`（待 R10）。仓库状态不变，重建断点仍在 **R2**。

## 2026-06-08 — ✅ v4 完美收官：100% 成功率，全程 3M 无崩

- **v4(单线程 + 几何碰撞，3M 步)干净跑完**：env steps 3.01M、60 次 eval、"saved final model"，**零崩溃**——
  线程超额订阅根治确认(越过 740k/1.45M/1.1M 三个历史崩点)。eval 平台 ~546。
- **v4 best 评估(满 60 密度，100 集)：SUCCESS 100%、0 碰撞、0 出界、0 超时，mean_return 544±25。**
  → **R8 完成，地基重建 R2–R8 全部收官。简化控制器完美穿越小行星带。**(旧地基 v1 4%/v2 7%/v3 发散 → 现 100%)
- 产物：`logs/ppo_rebuild_v4/best/best_model.zip`(主力模型)。v2(96%)亦保留。崩溃的 v1/v3 为残缺 run，可删。
- **下一步**：R9 键盘飞控(力/推进器模式) 或 R10 真实 17 推进器训练(`--dynamics realistic`，env 双模式已稳)。

## 2026-06-08 — 崩溃真凶 = 线程超额订阅（单线程钉死）→ v4

- **v3 也崩**(~1.1M)，但禁用接触后从"段错误"变成 `AttributeError: module 'numpy' has no attribute 'libc6'`
  ——发生在 eval env 的 reset→`sample_belt`。numpy 属性名被踩成 "libc6"(像 libc.so.6 路径垃圾)= **内存损坏**。
  `sample_belt` 每次 reset 逻辑相同、跑几千次才崩 → 非确定性损坏，非代码 bug。
- **根因定位**：16 SubprocVecEnv worker × (MuJoCo OpenMP + numpy BLAS 线程) + 主进程 torch 线程，在 24 核上
  **严重线程超额订阅** → 罕见跨线程内存踩踏。v1/v2 表现为段错误、v3 表现为 numpy 属性损坏，**同一根因**。
  (禁用接触/spawn 只是改变了损坏落点，没除根。)
- **修复**：`train_ppo.py` 顶端(import numpy/torch/mujoco 之前)设 `OMP/OPENBLAS/MKL/VECLIB/NUMEXPR_NUM_THREADS=1`
  + `torch.set_num_threads(1)`；spawn 子进程继承。每进程单线程 = 去竞争 + 提吞吐(常规 SB3+MuJoCo 向量化做法)。
- 几何碰撞检测(v3 引入)保留——本就干净、与线程修复正交。启动 **v4**(3M)验证全程无崩。

## 2026-06-08 — R8 突破：v2 达 96% 成功率 + 段错误根治 → v3

- **v2 best 模型评估(满 60 密度，100 集)：mean_return 530，SUCCESS 96%，0 碰撞，4% 出界，0 超时。**
  → reward 重平衡彻底奏效，简化控制器**学会穿越小行星带**(对比旧 v1 4% / v2 7% / v3 发散)。
  eval 曲线 −128→+3(500k)→+283(900k)→+547(1.15M)→+545(1.4M)，清晰上升。
- **但 v2 仍在 ~1.45M 步段错误**(spawn 没根治，v1 740k→v2 1.45M)。根因确认：**MuJoCo 接触求解器**——
  飞船学会高速前冲后，高速撞上轻质 free-joint mesh 陨石产生能量化接触，在 C 层炸；python 守卫拦不住 mj_step 内部。
- **根治**：禁用飞船↔陨石的**物理接触**(belt_generator 把陨石 contype/conaffinity 设 0，陨石只漂移不产生接触)，
  碰撞改**几何检测**(env `_capsule_collision`：飞船胶囊 body-X 段 vs 陨石保守外包球 r_eff，surf≤0 即撞)。
  MuJoCo 永不求解能量化接触 → 段错误源消除。几何检测偏保守(早一点)、安全。check_env 双模式过(随机 19/20 碰撞)。
- 启动 **v3**(`logs/ppo_rebuild_v3`，3M)验证：须越过 1.45M 不崩。v1/v2 保留对照。

## 2026-06-08 — R8 训练 v1 崩溃 + reward 重平衡 → v2

- **训练 v1**(`logs/ppo_rebuild_v1`，16 envs，课程 8→60)：~740k 步 **MuJoCo 段错误(core dumped)** 崩溃
  (conda 包装把退出码显示成 0，实为 segfault + worker 内存污染连带的 `_rand_quat` 赋值 ValueError)。
  单进程 800 集压不出 → 判定为 fork+MuJoCo+torch 下的罕见 race。
- **v1 学习也失败**：eval(满 60 密度)全程走平 −145，best 模型 100 集 = **0% 成功 / 90% 出界 / 0 碰撞**。
  诊断：**reward 失衡**——proximity 惩罚每步最高 0.4×12=4.8 ≫ 进度奖励每步 ~0.3，策略学会"远离所有石头"
  宁可侧漂出界也不穿越(v1/v2 同类病重现)。
- **修复**：①抗崩溃：SubprocVecEnv 改 `start_method="spawn"`(规避 fork race)+ reset 四元数赋值强制
  `np.asarray(...,float)` + step 加非有限值守卫(qpos/qvel/reward/obs 非有限→当碰撞终止/nan_to_num)。
  ②reward 重平衡：w_proximity 0.4→**0.05**(软提示，collision −300 才是硬威慑)、w_dist 1.0→**2.0**(强进度)、
  g_safe 6→8 / w_gload 0.5→**0.15**(别压死推力)。实测温柔巡航 +0.04/步(进度主导)。
- 启动 **v2**(`logs/ppo_rebuild_v2`，3M 步，同课程)。v1 保留供对照。待中途 eval 验证学习。

## 2026-06-08 — 地基重建 R2–R7 落地（土豆带 / 重摆 / 雷达 / F8C 力标定 / G-load 奖励）

- **R2** `belt_generator.py` 重写：free-joint 土豆 mesh 陨石(每颗独立 mesh 资产=随机库网格×幂律尺寸×各轴
  aspect+随机朝向)、最小间距防重叠、build-once。返回 `list[Asteroid]`(body/geom/joint 名 + r_eff 保守外包球)。
  冒烟：60 颗全摆下，幂律生效(67% 低于均值，r_eff 2–12m)。
- **R3** env reset 改重摆：写 qpos(位置+随机朝向)/qvel(慢漂移+自转)，不重编译。课程密度 = `n_active` 激活子集
  + 其余陨石停泊到 x=5000 界外(模型固定大小、密度可变)。
- **R4** obs 改**全向球面雷达**：体系 12×6=72 格×2 通道(proximity 1/(1+surf) + 径向接近速度)，无遮挡解析分桶、
  同格取最近、保守外包球 r_eff。obs_dim 56→**160**。
- **R5** 力按**官方 F8C 数据**标定(用户提供，存 memory `f8c-performance-specs`)：sign-asymmetric 映射
  (action=0→0力)前推 mass×103.5(10.55G)/反推 mass×36.3(3.70G)/侧移 mass×40；力矩按 ~2s 达官方转速标定
  (roll140/pitch38/yaw35°/s)，惯量经 `body_iquat` 转回 body 系取对角(避免主轴重排取错)。XML 6 虚拟执行器
  ctrlrange 拓宽为不夹断包络。**实测加速度/转速全部精确命中**(103.5/-36.3/40 m/s²，140/38/35°/s)。
- **R6** reward 加 **G-load 惩罚**(超 g_safe=6G 按差扣分，保飞行员命——飞船能 10.55G 但飞太猛要罚)；结构=
  potential+heading+proximity(d_safe12)+spin+gload+ctrl/time+collision300/success200/oob100。实测满油门单步
  −2.23、温柔 5G 单步 +0.04。**权重数值留 R8 按失败模式微调**。
- **R7** 工具：preview_belt 改用 env 显示漂移+自转；check_env 双模式过；rollout/eval 兼容(randomize_belt 留为
  忽略参数，随机带由 reset 重摆)。
- **check_env 双模式均过**(obs 160)。随机动作碰撞率高(17/20)= 力变真实、不再瞬移，符合预期。
- 改了 `environment.xml`(虚拟执行器 ctrlrange 拓宽)、`CLAUDE.md`(RL stack 段) 、REBUILD_TODO(勾 R2–R7)。
- **下一步 R8**：冒烟训练 → 课程 PPO 正式训练(16 envs / n_start→60 / 3-5M 步) → eval_policy 看成功率。

## 2026-06-08 — R4 obs 改雷达式（设计定稿，未实现）

- 用户决定把 obs 从 K-最近邻改成**雷达式**。设计敲定（写进 REBUILD_TODO R4）：
  体坐标系全向球面雷达，方位×俯仰格(默认 12×6=72)，每格 2 通道(表面距 + 径向接近速度)。
  **无遮挡**(解析分桶，近不挡远)、**同格取最近**、陨石按**保守外包球** r_eff(scale×网格外接半径)处理。
  纯 numpy 实现不用 mj_ray。obs_dim = 16 + 2·N_az·N_el(默认 160)。
- 尚未写代码，断点仍在 R2。

## 2026-06-08 — 重命名 conda env → asteroid-belt-runner

- 用户要求把 RL env 由 `space-robotics-project` 改名为 **`asteroid-belt-runner`**。
  `conda rename -n space-robotics-project asteroid-belt-runner`（clone+remove，旧 env 已删）。
  验证：`conda run -n asteroid-belt-runner` 下 mujoco 3.3.7 / CUDA True。
- 全仓库批量替换 `space-robotics-project`→`asteroid-belt-runner`（11 文件：所有 Agent_tool 脚本、
  train_ppo.py、asteroid_mesh.py、CLAUDE.md、PROJECT_PLAN.md、REBUILD_TODO.md、本日志历史条目）。
- **命名碰撞处理**：README quick-start 本就用 `asteroid-belt-runner`(python3.11+numpy2.x 旧手动配置)，
  与新 RL env 同名。改 `CLAUDE.md` Setup 段：声明统一为单一 env(python3.10/numpy<2，RL+试玩共用)，
  标注 README 那套 create 配方已过时、勿用。README 本身暂未改（仅一处 quick-start，留待 R10 文档统一）。

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
