# Claude Code 操作日志

记录 Claude Code 在本仓库的实质性操作，便于跨会话 / 跨平台接续。
规约见 `Agent_log/README.md`（每次实质操作后追加，不记录用户提问）。

---

## 📌 现状速览（读这段就够 · 2026-06-08）

- **成果**：简化动力学控制器**学会穿越大陨石带** —— v15 = **SUCCESS 41% / 出界 0% / 碰撞 59%**（满密度 100 集）。
  主力模型 `models/ppo_v15_best_41pct.zip`（=v15 的 1.5M checkpoint；后段发散，故手动挑峰值）。
- **环境**：conda env **`asteroid-belt-runner`**（**Python 3.10** + MuJoCo3.3.7 / Gym0.29.1 / SB3 2.3.0 /
  torch2.2.1+cu121 / numpy<2 / imageio）。换机器照 `CLAUDE.md` Setup 段重建。
- **崩溃真凶（最终定论）**：torch 2.2.1 的 `torch._dynamo` **import 时偶发崩**（~15%，3.10 报 sre_compile / 3.11 段错误，
  与 Python 版本无关）。本只在 import 期、基本无害。**整场崩溃马拉松是我自找的**：我曾显式给 SubprocVecEnv 设
  `start_method="spawn"`，**覆盖了 SB3 的安全默认 `forkserver`** → 16 个 worker 各自重 import torch → 16 倍触发。
  **修复 = 别覆盖**（用默认 forkserver）。用户用默认配置 train 从不崩。手动试玩/单进程评估永不触发。
- **带最终定型**（`belt_generator.BeltConfig` 默认）：n=40（带内~31）、min_gap=55（缝~58m，对 26m 盒子碰撞飞机
  ~32m 余量）、半径120×长400、goal x=540、盒子碰撞贴 STL（机翼算）、oob_margin=25、廉价 reset（预算布局+绕X旋转）。
- **任务最新改造**：随机偏轴出口（每回合 (540,gy,gz)，偏轴 40–90m）+ 朝目标速度奖励 + 可见绿色出口光球。
  **代码就绪、check_env 过，但尚未训练这个新任务**。
- **下一步**：① 训练新任务（随机出口）；② 治后段发散（早停/降LR）降碰撞；③ R9 键盘飞控；④ R10 真实17推进器训练。
- **关键约定/警告**：长训用 `Agent_tool/train_resilient.sh`（崩了自动 --resume）；回放 `Agent_tool/watch.sh`。
  **train 别动 SubprocVecEnv 的 start_method**（默认 forkserver 才对；spawn=重 import 风暴，fork=CUDA 死锁）。

---

> 以下为按时间顺序的详细流水账。注意 v5–v14 期间对"崩溃原因"有过多次**错误归因**（线程超额订阅、MuJoCo 接触求解器、
> TorchDynamo env-var…），最终定论见上方速览（spawn 覆盖默认）；保留原记录以存调试历程。

## 2026-06-08 — 项目脚手架 + 总体计划落地

- **确认运行环境**：本仓库 RL 用 conda env（Python 3.10），已装 MuJoCo 3.3.7 / Gymnasium 0.29.1 /
  Stable-Baselines3 2.3.0 / PyTorch 2.2.1+cu121 / pynput。缺 trimesh（仅 `STL_inertia_calculator.py` 用）。
  此 env numpy=1.26，与根目录 `requirements.txt` 写的 2.3.3 不一致 —— RL 以本 env 为准。
- **建立目录约定（借鉴 Human_exo_interaction）**：新建 `Agent_tool/`（一次性小工具）、`Agent_log/`（本日志，
  git 跟踪并 push）、`TODO_list/`（计划）、`logs/`（训练输出，git 忽略）。
- **改 `.gitignore`**：忽略 `logs/ runs/ checkpoints/ MUJOCO_LOG.TXT *.zip wandb/` 等训练产物，但 `Agent_log/` 保持跟踪。
- **写总体计划** `TODO_list/PROJECT_PLAN.md`：6 阶段路线图（脚手架→小行星带→简化RL环境→训练→真实17推进器→真实训练）。
- 现有 `main.py` 的直接速度控制（写 `data.qvel`）定位为"试玩"入口，不参与 RL；RL 动作用 XML 里的 6 个力/力矩执行器。

## 2026-06-08 — Phase 1+2+3 落地（小行星带 / RL 环境 / 训练管线）

- **Phase 1** `envs/belt_generator.py`：`mujoco.MjSpec` 运行时组装场景。从 `environment.xml` 加载 → 飞船加隐形胶囊
  **碰撞代理**（STL 只做可视）→ +X slab 内 rejection-sample 散布 N 颗球形小行星。碰撞掩码：飞船 vs 小行星相撞，
  小行星互不撞、不碰坐标轴。支持 `dynamic=True` 漂移。冒烟 nbody=3 ngeom=65。
- **Phase 2** `envs/asteroid_belt_env.py`：`AsteroidBeltEnv`。obs 56 维（v3+ω3+fwd3+up3+goal_dir3+dist1+最近K=8×5），
  action 6 维→执行器 ctrlrange。reward=+X 进度−控制−时间，碰撞−100/到达+200/越界−50。**踩坑**：MjSpec 存成
  `self.spec` 与 Gym 保留属性冲突 → 改 `self.mj_spec`。`check_env.py`：env_checker 过 + 20 集随机 rollout。
- **Phase 3** `train/train_ppo.py`：SB3 PPO + SubprocVecEnv(8) + Checkpoint/Eval + tensorboard。冒烟训练通过（~5000fps）。

## 2026-06-08 — 正式训练 v1 + Phase 4 真实动力学 + reward 翻车与修复

- **训练 v1**（2M 步）：eval 回报 −3→100~135 但 `eval_policy` 100 集 = success 仅 **4%**、oob **79%**。诊断：
  x-progress reward 可被"猛冲 +X 刷分后侧飞出界"刷满 → reward 设计缺陷。
- **修复（v2）**：改 potential-based `w_dist*(prev_goal_dist−goal_dist)` + heading；oob 50→100、ctrl_cost 0.01→0.001。
- **Phase 4 真实动力学** `envs/thruster_layout.py`：17 推进器（2主+X / 3反向 / 12 RCS），每个=site+site-actuator。
  **wrench 矩阵 rank=6、6-DOF 可控**。`dynamics` 开关切简化(6)/真实(17)。check_env 双模式过。
- **新工具** eval_policy / rollout_viewer / preview_belt。启动 v2（potential reward）。

## 2026-06-08 — v2 评估 + reward 再迭代（避障）+ 课程学习 → v3

- **v2**（100 集）：success 7%、collision **72%**、oob 21%。修好了侧飞但变成**直冲撞石头**（无提前避障激励）。
- **v3 参数**：加 proximity 惩罚（w 0.4 / d_safe 12）+ spin 惩罚 0.01；collision_penalty 100→**300**。
- **课程学习**：env `set_n_asteroids()` + train `CurriculumCallback`（前 60% 密度 n_start→n_end）+ `--curriculum`。启动 v3。

## 2026-06-08 — 用户叫停、地基重建（决策已定）

- **诚实评估**：管线齐全但控制器没学会（success 4→7%，v3 发散到 −221 已 kill），地基占位（圆柱走廊塞随机球、
  尺度拍脑袋、简化力瞬间机动）。用户认同要重建，叫停迭代。
- **用户三项决策**：①尺度=游戏化标度；②RL 动力学=简化6力先行（力调合理量级、非瞬间机动）；③小行星=缓慢漂移+自转。
- **小行星带按"结构真实、密度调高"重建**（用户授权自行设计）：幂律尺寸、不规则外形+随机朝向、低速漂移自转、最小间距。

## 2026-06-08 — 换机器：重建 conda 环境 + 重命名

- **新机器**：原 env 不存在（只有 base / myoassist），GPU = RTX 4090 (24G)。重建 `conda create python=3.10` + pip 装
  mujoco3.3.7/gym0.29.1/sb3 2.3.0/numpy<2/torch2.2.1(cu121)/pynput/tensorboard。check_env 双模式过、CUDA True。
- **重命名 env**：用户要求 `space-robotics-project` → **`asteroid-belt-runner`**（conda rename，clone+remove）。
  全仓库批量替换该名（Agent_tool 脚本、train、CLAUDE.md、PROJECT_PLAN、REBUILD_TODO、本日志历史条目）。
  处理命名碰撞：CLAUDE.md Setup 声明统一为单一 env、标注 README 旧 create 配方过时勿用。

## 2026-06-08 — R4 obs 改雷达式（设计定稿）

- 用户决定 obs 从 K-最近邻改**雷达式**：体系全向球面雷达，方位×俯仰格（默认 12×6=72），每格 2 通道（表面距 +
  径向接近速度）。无遮挡解析分桶、同格取最近、陨石按保守外包球 r_eff。纯 numpy。obs_dim = 16 + 2·N_az·N_el（默认 160）。

## 2026-06-08 — 地基重建 R2–R7 落地（土豆带 / 重摆 / 雷达 / F8C 力标定 / G-load）

- **R2** `belt_generator.py` 重写：free-joint 土豆 mesh 陨石（独立 mesh×幂律尺寸×各轴 aspect+随机朝向）、最小间距、
  build-once。返回 `list[Asteroid]`（body/geom/joint 名 + r_eff 保守外包球）。
- **R3** reset 改重摆：写 qpos（位置+随机朝向）/qvel（慢漂移自转），不重编译。课程密度 = `n_active` 激活子集 + 其余停泊界外。
- **R4** obs 改**全向球面雷达**（12×6×2 通道）。obs_dim 56→**160**。
- **R5** 力按**官方 F8C 数据**标定（存 memory `f8c-performance-specs`）：sign-asymmetric 映射前推 mass×103.5(10.55G)/
  反推 mass×36.3(3.70G)/侧移 mass×40；力矩按 ~2s 达官方转速（roll140/pitch38/yaw35°/s），惯量经 `body_iquat`
  转回 body 系取对角。XML 6 虚拟执行器 ctrlrange 拓宽。**实测加速度/转速精确命中**。
- **R6** reward 加 **G-load 惩罚**（超 g_safe 按差扣，保飞行员命）。结构=potential+heading+proximity+spin+gload+ctrl/time+
  collision300/success200/oob100。
- **R7** 工具更新；check_env 双模式过（obs 160）。

## 2026-06-08 — R8 训练 v1 崩溃 + reward 重平衡 → v2

- **v1**（16 envs，课程 8→60）：~740k 步崩溃（当时记为"MuJoCo 段错误"，**实为 torch dynamo import 抽风**）。
  学习也失败：best 100 集 = **0% 成功 / 90% 出界**。诊断 reward 失衡——proximity 每步最高 4.8 ≫ 进度 ~0.3，
  策略学会"远离所有石头"宁可出界。
- **修复**：①〔当时误判〕改 `start_method="spawn"`（**这一步其实是埋下后患的根源**，覆盖了 SB3 默认 forkserver）+
  reset 四元数强制 float + step 非有限值守卫；②reward 重平衡 w_proximity 0.4→**0.05**、w_dist→**2.0**、g_safe 6→8 /
  w_gload 0.5→**0.15**。启动 v2。

## 2026-06-08 — R8 突破：v2 达 96% + 段错误（误判→禁用接触）→ v3

- **v2 best（满 60 密度 100 集）：SUCCESS 96%、0 碰撞、4% 出界。** reward 重平衡奏效，**学会穿越**（eval −128→+547）。
- 但 v2 仍 ~1.45M 崩（当时**误判**为 MuJoCo 接触求解器：高速撞 free-joint mesh 陨石能量化接触在 C 层炸）。
- 〔基于误判的改动，但本身干净、保留〕禁用飞船↔陨石**物理接触**（陨石 contype/conaffinity=0），碰撞改**几何检测**
  （`_capsule_collision` → 后改 `_box_collision`）。启动 v3。

## 2026-06-08 — 段错误（误判→线程钉死）→ v4

- **v3 也崩**（~1.1M），禁用接触后从段错误变 `numpy has no attribute 'libc6'`（内存损坏样）。
- 〔当时**误判**根因为线程超额订阅〕修复：train_ppo.py 顶端设 `OMP/MKL/...=1` + `torch.set_num_threads(1)`。
  （此修复本身是常规好做法、保留；但**不是真崩因**。）启动 v4。

## 2026-06-08 — ✅ v4 完美收官：100%（但其实是"绕飞"作弊带）

- **v4 干净跑完 3M、零崩溃**，best 100 集 = **SUCCESS 100% / 0 碰撞**。当时以为大功告成。
  （事后看：v4 这次没崩是 spawn 下"运气好"没触发那 15%；100% 是钻了"绕行走廊"的空子，见下条。）
- 产物 `logs/ppo_rebuild_v4/best/best_model.zip`。

## 2026-06-08 — 发现 v4 在"绕飞"作弊 + 加宽带 → v5/v6

- **用户目视 v4 viewer 发现飞船绕带外缘飞、没真穿越**。诊断坐实：带半径 45，飞船穿越时平均侧偏 72（远在带外），
  出界边界宽到 105 留了绕行走廊 → 100%/0 碰撞是钻空子。
- **修带几何**：belt_yz_radius 45→55、出界余量 60→12（新增 `oob_yz_margin`，堵死绕行）、n 60→90。`watch.sh` 保存。
- **v5（新带）~450k 又崩**——〔仍未识别真凶〕判定"与其根治不如能恢复"。**抗崩溃训练**：train_ppo.py 加 `--resume`，
  `Agent_tool/train_resilient.sh` 包装重试循环。启动 v6。

## 2026-06-08 — 飞机朝向修正 + 大带重设计（可视化迭代）→ v7

- **飞机朝向 bug**：用户发现机头朝绿轴(+Y)，但 env 以 +X 为前向 → 飞船"侧着飞"。**踩坑**：用 `euler` 修被 MuJoCo
  mesh 内部帧误导。改用**离屏渲染(EGL)+ 真实矩阵 `data.geom_xmat`** 验证：正确解 = 绕 Z 转 −90°（`quat="0.7071 0 0 -0.7071"`，
  pos 同步转正）。机头→+X(红)、翼→Y(绿)、顶→+Z(蓝)。纯视觉。
- **大带重设计**（用户驱动，反复 preview）：半径 120 × 长 400、目标 x=540。密度按"间距 ≥ 飞机尺寸倍数"——用户最终选
  稀疏宽缝。
- 启动 v7（230 颗）。`environment.xml` 加 geom quat/pos（朝向）。

## 2026-06-08 — 缝隙/碰撞体诊断 + 廉价 reset（v8/v9 → v10）

- **v5–v8 反复学不会**。诊断 v8 best（90颗大石头）：0%成功、70% 侧向出界、平均只飞到 x=127（进带口就垮）。根因带堵死：
- **缝隙诊断**（用户提出）：最近邻表面间隙**中位仅 13m < 石头直径 27m**，48% 缝 < 飞机碰撞径——钻不过近半窄缝。
  根因 `min_gap` 仅 1.5m → **min_gap 1.5→32**。
- **碰撞体改盒子**（用户问"碰撞球怎么算、能否贴 STL"）：原 12m 胶囊忽略机翼 → 改贴紧 **OBB 盒子**（half-extents
  12.21/12.83/2.93），机翼算碰撞（`_box_collision`）。飞机有效宽 12→26m。配套 oob_margin 12→**25**。
- **廉价 reset**：min_gap 越大 `sample_belt` 每 reset 拒绝采样越疯（**当时仍误判为崩因之一**）→ build 时预算布局、
  reset 只挑一套 + 绕 X 旋转（O(N)，1.3ms，快 ~50×）。固定 body 数。启动 v10。

## 2026-06-08 — ⚠️ 抓到真凶 torch dynamo（仍未完全识破 spawn）

- v11 抓到完整 traceback：崩溃在 `torch._dynamo/skipfiles.py → re.compile → sre_compile`，报 `too many values to unpack`。
  **torch 2.2.1 的 dynamo import 偶发抽风**，和陨石/MuJoCo/我代码无关。之前 v5–v10 归因全错（内存/陨石/线程）。
- 〔当时方案〕`TORCHDYNAMO_DISABLE=1`（无效，挡不住 import）→ 后续又试 eager import、移进 main、warm-import 重试…
- **顺手做的真改进保留**：min_gap、盒子碰撞、廉价 reset、固定 body 数。

## 2026-06-08 — 崩溃彻底根治（forkserver）+ 大带学会穿越（41%）

- **关键纠错（最终定论）**：崩溃 = torch 2.2.1 dynamo import 偶发崩；被我**显式 `start_method="spawn"`**（覆盖 SB3 默认
  forkserver）放大成"16 个 worker 各重 import"→ 狂崩。期间试过 spawn（狂崩）/ fork（fork-after-CUDA 死锁 hang）/
  **forkserver（对）**。修复三连：①start_method 用 **forkserver**（后又简化为：删掉显式设置、直接吃 SB3 默认）；
  ②torch/SB3 移进 `main()`（worker 不碰 torch）；③`_warm_import` 重试 + wrapper 兜主进程启动偶发。
- **带最终定型（为可学放宽）**：n=40（带内~31）、min_gap **55**（缝~58m、对 26m 飞机 ~32m 余量）。之前 min_gap32（缝仅
  4m 余量）飞机挤不过 → 0%。
- **v15 训练**（课程 8→40，3M，forkserver）：**1.5M checkpoint 达峰 SUCCESS 41% / 出界 0% / 碰撞 59%**（100 集）——
  真从缝里穿、不再钻空子。**后段发散**（2M→0%，典型 PPO 不稳）；resume 重置了 EvalCallback best，故手动挑 1.5M
  checkpoint 存为主力。`logs/ppo_rebuild_v15/best/best_model.zip` + `models/ppo_v15_best_41pct.zip`。

## 2026-06-08 — 保存收尾 / 跨机器提交

- 用户满意（看了回放），要求保存换机器。提交（分支 `rebuild/asteroid-belt-rl`）：`10ef9be` 大带重设计+41%+forkserver、
  `59baeec` 跟踪 v15 模型。
- **主力模型入 git**：`models/ppo_v15_best_41pct.zip`。踩坑：`.gitignore` 的 `*.zip` 忽略了它，加例外 `!models/*.zip`
  ——`.gitignore 不支持行内注释`，注释挪单独一行才生效。+ `models/README.md`。更新 REBUILD_TODO 断点段。
- 推送：本环境无 git 凭据，用户自行 `git push`。

## 2026-06-08 — 任务改造（随机出口+求快） + Python 3.11 根治尝试（失败）

- **任务改造**（用户加难度）：①**随机偏轴出口**——每回合目标 `(goal_x, gy, gz)`，gy/gz 偏轴 40–90m；势能/obs/heading
  指向 3D 出口；成功=进出口点 `goal_radius=25m` 球（逼绕到指定出口、不能直冲）。②**求快**——`w_speed=0.02×(速度·目标方向)`
  + time_cost 0.02→0.03。check_env 过、随机出口生效。
- **Python 3.11 根治尝试 → 失败**：建 abr311 测 `import torch._dynamo`，0/20（运气）→ 改名后 4/20，且 3.11 里直接段错误。
  **3.11 没修好**（torch 2.2.1 C 层 import 抽风与 Python 版本无关）。根治只能换 torch 2.3+（兼容风险）。

## 2026-06-08 — 真凶定论（spawn 覆盖默认）+ 换回 3.10 + 出口标记 + 用 SB3 默认 + 整理日志

- 用户追问"为什么会崩、我自己 train 也没崩过"。查 SB3 源码：**`SubprocVecEnv` 默认 start_method 在 Linux 上就是
  `forkserver`**。**整场崩溃是我自找的**——我曾显式设 `start_method="spawn"` 覆盖了这个安全默认。用户用默认 → 永不崩。
- **代码改最干净**：train_ppo.py **删掉显式 `start_method`**，直接吃 SB3 默认（forkserver）。加注释警告别覆盖。冒烟过。
- **换回 Python 3.10**（3.11 无好处）：重建 `asteroid-belt-runner`=py3.10+同栈+imageio，check_env 过。CLAUDE.md/README
  改回 3.10，崩溃说明改成真相（spawn 覆盖默认）。
- **加可见出口标记**：build_scene 加半透明绿球 `goal_marker`（无碰撞），reset 时移到随机出口（`model.geom_pos`），
  viewer 里能看到该飞到哪。
- **整理本日志**：加顶部"现状速览 TL;DR"+ 把被 prepend 打乱的条目按时间顺序重排（内容保留，标注 v5–v14 的错误归因历程）。
