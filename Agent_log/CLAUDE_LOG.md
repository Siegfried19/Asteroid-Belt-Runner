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

## 2026-06-08 — 新机器：发现 rebuild 分支未合并，ff 合入 main

- 用户换新机器(i7-14700F/31G/RTX4070S)，clone 后只在 `main`(旧球体地基)上工作。
- 我先误在旧 main 上做了"小行星带三维放大"(球体场 n=900 等)——**建在过时地基上，已全部 `git checkout` 丢弃**。
- 用户目视 preview 发现"石头太小/全圆球/飞机朝向不对"，怀疑没 merge。**确认属实**：
  真正的地基重建全在 `origin/rebuild/asteroid-belt-rl`(领先 main 7 commit)，从未合回 main。
- merge base = main 的 HEAD → **fast-forward 合并，零冲突**。main 现含：幂律尺寸/土豆 mesh
  外形+随机朝向/free-joint 漂移+自转/飞船 geom `quat` 朝向修正/160维全向雷达 obs/F8C 标定推力/
  build-once 重置/G-load 惩罚/v15 best 模型(41%)。
- 新机器仅有 `space-robotics-project` env(无文档所述 `asteroid-belt-runner`，技术栈相同)。
  用它跑 `check_env`：obs=160，两种动力学通过，随机策略出现碰撞(石头够大够密，合理)。
- **待办**：conda env 名与文档(`asteroid-belt-runner`)不一致，需统一(建新 env 或改文档)。

## 2026-06-08 — 重建 conda 环境(asteroid-belt-runner)+ 渲染验证新场景

- 用户要求建新环境 `asteroid-belt-runner`、删老的 `space-robotics-project`、并查看新场景。
- **磁盘真相**：`/home` 是独立分区 `/dev/nvme1n1p4` **仅 117G 且 100% 满**(anaconda3 占 57G，
  内含 myoassist×3/myosuite/RL-test/Human-exo 等 6-8G 环境)。`conda create --clone` 因
  `No space left` 失败(需双份空间)，残留 6.1G 不完整残骸已删。
- **改用 `mv` 改名**(瞬间、零额外空间)：`envs/space-robotics-project` → `envs/asteroid-belt-runner`。
  conda 立即识别；`check_env` 用新环境名通过(obs=160，两动力学 OK)。
- mv 副作用：`bin/` 下 29 个脚本 shebang 仍指旧路径，已 `sed` 批量修正(pip 等可直接用；
  `conda run ... python` 本就不受影响)。环境名现与文档(CLAUDE.md)一致。
- **新工具 `Agent_tool/render_scene.py`**：EGL 离屏渲染场景到 `images/scene_*.png`(overview/
  ship_side/ship_top/downaxis)。两个渲染坑：①离屏 framebuffer 默认 640→设 `m.vis.global_.offwidth/height`；
  ②大 extent 把 znear 推远导致近距特写全黑→设 `m.vis.map.znear=0.002`。
- **目视确认用户三问题全解决**：小行星=不规则土豆 mesh+幂律尺寸(非小圆球)；动态散布在整个 YZ 圆截面
  (3D 体积非细管)；飞船机头朝 +X、机翼沿 Y 对称无偏航(quat 修正生效)。

## 2026-06-09 — 放大+大偏离 RL 攻坚:6 轮训练 + 根因诊断(冲撞局部最优,未训通)

用户授权自主推进。把小行星带放大(600m×180m×135颗,密度不变)+ 出口偏离加大(exit_r 40-90→90-150),
接 curriculum(密度+偏离同步爬)后正式训练。**结果:整套"最终架构"从未训通,根因是冲撞局部最优。**

**训练/评估全记录(满难度评估,deterministic,100 集):**
| run | 配置 | success | 失败分布 |
|-----|------|---------|----------|
| v16 | 原reward, 放大+大偏离 | 0% | 84%撞/16%oob |
| v17 | reward修复(w_dist2→1,coll300→900,prox0.05→0.6,succ→400,oob→200,gr25→35) | 1% | 87%撞 |
| v18 | 易curriculum(n5→90,exit0→150,ramp0.7,ent0.01,3M) | 1% | 67%撞/32%oob |
| baseline | 原版难度n40/exit40-90/原reward,2M | 2% | 63%撞/35%oob;eval早期974→末期208退化 |
| **v15历史模型@当前env** | — | **2%** | 70%撞/28%oob(README记41%/oob0%) |
| stage1 | 我的reward, n80放大, **exit_r=0**(直线), 3M | **0%** | 68%撞/32%oob |

**根因诊断:**
- **action=0(飞船不动)→ 20/20 timeout**:env 无"强制撞"bug,停着安全。是策略**主动**冲撞。
- 训练 eval reward 反复收敛到 **-297**;而"不动"≈-66。**PPO 学到比不动更差的策略**=典型**冲撞局部最优**:
  progress 奖励在冲的过程中持续上升(稠密梯度),把策略推向"全速冲",越过避障悬崖撞墙。困在"冲"吸引盆。
- **机动性充足**(fwd 103.5/lat 40 m/s²),非物理瓶颈。
- **v15 的 41%(及 v4 的 100%)是 rebuild 早期、更简单架构(直线穿越、无 off-axis exit、不同 reward)的成绩。**
  commit 654bdb3 引入"random off-axis exit + speed reward"后,当前"最终架构"(雷达 obs/F8C 推力/off-axis)
  **搭好但 wrap-up 时从未训通**。README 的 41% 是过时架构的旧数据。

**诊断 gap(诚实记录):** 未干净复现"原 reward + exit_r=0 + 原场景 n60"(最接近 v4/v15 可学配置)——
stage1 的 exit_r=0 混入了我的 reward + 放大场景。**这应是后续攻坚第一步**,以区分"我的 reward 的锅"vs"env 回退"。

**攻坚方案(留待后续):**
1. 干净复现 v4/v15:原 reward + exit_r=0 + 原场景,确认早期可学性是否还在。
2. 打破冲撞局部最优:**去 speed reward(w_speed=0)** + **限飞船最大速度**(冲太快没时间避障) +
   **progress 稀疏化/封顶**(别沿途刷分) + **ent_coef↑**(跳出"冲"吸引盆)。
3. 验证能打破后,阶梯加 off-axis exit(0→30→60→90→150)找"高通过率 vs 偏离"边界。

**当前工作区状态(已保留,未 commit):** 放大+大偏离+reward rebalance+curriculum(密度&偏离)+
render_scene.py + eval_policy(--exit-r/--max-steps) + preview 改进。logs/ 下 v16/v17/v18/baseline/s1 为
诊断产物(git-ignored)。

## 2026-06-09 — RL 攻坚(续):破"冲撞局部最优",net512 是关键,航程 curriculum 待训

完整 ~25 轮实验见 `Agent_log/REWARD_EXPERIMENTS.md`(自主迭代台账)。要点:
- **破解冲撞局部最优**:重设计 reward——去 speed reward(冲撞元凶)、closing 逼近惩罚做避障主力(只罚朝石头冲、
  绕开不罚)、二次接近惩罚(窄 d_safe 不堵间隙)、anti-retreat(堵倒退逃跑)。
- **真正瓶颈是网络容量,不是 reward**:reward/ent/VecNormalize 调了 20 轮卡 22%;**net512(512²)+4M 步直接到
  极简 100%、n40 短航程 72%**。教训:学不会先怀疑网络容量/训练量,别死磕 reward。
- **当前障碍=航程断崖**(航程 300→500 飞船恐惧倒退、0%):已实现**航程 curriculum**(`env.set_traverse` +
  CurriculumCallback ramp goal 距离 + `--traverse`),冒烟验证 OK,**待正式训练**。
- train_ppo 新增能力:`--net-width`(网络宽)、`--vecnorm`(reward 归一化 flag,默认关)、`--traverse`(航程 curriculum)、
  curriculum 现 ramp 密度+偏离+航程三维。env 新增 reward 项:closing/proximity 拆分、arrival(关)、anti-retreat。
- **env 默认是调试中间态**(goal_radius 60 放宽等);下次从 `REWARD_EXPERIMENTS.md` 末尾 "下次 TODO" 继续。
- 训练产物 `logs/ppo_*`(git-ignored);最佳极简模型 `logs/ppo_diag_net512`(100%)。

## 2026-06-09 — 换新机器(i9-14900K/64G/RTX4090),接航程 curriculum TODO 起训

- **新机器**:i9-14900K(32线程)/64G/RTX 4090 24G/73G 空闲——比上台(i7-14700F/RTX4070S)强。
  `asteroid-belt-runner` env 已在;`check_env` 过(obs=160,两动力学 OK,随机策略多 timeout/少撞,合理)。
- **logs/ 只剩旧 run**(v2/v4/v11/v15):net512/diag_net512/n40 是 git-ignored,没随仓库过来→需重训。
- 确认代码能力齐:`--traverse/--net-width/--exit-r-end/--n-start/--curriculum`、env `set_traverse`、
  eval_policy `--exit-r/--max-steps/--n-asteroids` 都在。
- **建 todolist + 起训 REWARD_EXPERIMENTS.md 末尾 TODO 第1步(破航程断崖)**:
  `--curriculum --traverse 300 700 --n-asteroids 40 --n-start 5 --net-width 512 --timesteps 4_000_000 --exit-r-end 0 0`。
- **首训失败(0% success / 73%撞 / 27%出界)→ 抓到 TODO 命令行 bug**:漏传 `--ent-coef`→默认 0.0,
  而 net512 突破靠 **ent0.05**(台账载 ent0.05→0.02 即 22%→6%)。满难度后段 1.6M 步 return 完全平(-890)=
  卡死"撞"局部最优,非训练不足。i9+4090 跑 4M 仅 ~28 分钟。
- **重起干净 A/B:只补 `--ent-coef 0.05`,余同**。详见 REWARD_EXPERIMENTS.md。

## 2026-06-09 🎉 航程断崖其实早破了——真凶是 best_model 选择 bug;满航程 n40=63%

- ent0.05 run 训完(~14 分钟),eval `best_model.zip` 仍 0%——但那是假象。**直接测 final `model.zip`**:
  **ent0.05 = 63% success / 37%撞 / oob 0% / timeout 0%;ent0 = 60%**。**航程断崖破了**(上台机器 85% oob 消失)。
- **根因**:EvalCallback `n_eval_episodes=10`、eval 方差 ±800 → best_model 纯运气选中 50K 早期噪声(-402),
  真正训好的 final 在自己 10 集上运气差(-951)→ 我们一直在测 50K 垃圾。**ent_coef 其实几乎无差**(63 vs 60)。
- **修复**:train_ppo.py `n_eval_episodes` 10→40(降 SEM);约定**永远同时测 final model.zip**。
- 主力模型存 `models/ppo_traverse_n40_63pct.zip`(当前真实架构历史新高;README 旧 41% 是旧简化架构)。
- 仅剩失败模式=碰撞 37-40%。下一步:降碰撞 或 推 curriculum 加难(密度/收球/off-axis exit)。

## 2026-06-09 — 推 n135 加难,踩到 build 端 numpy corruption(高密度放大);sqrt 修复

- **起 n135 curriculum(8M)→ 启动几乎必崩**:`sample_belt` 里 `np.linalg.norm` 抛
  `'Float64DType' object has no attribute 'dtype'`。n40 没事、n135 几乎每次——因为 sample_belt 拒绝采样在
  n135 下 norm 调用量是 n40 的几十倍(135×200×~100 placed),16 个 forkserver worker 并发 build 撞上 **C 层
  numpy 内存 corruption**(和当年 spawn 放大 torch flake 同理)。单进程少量调用不易现。
- **诊断**:单进程密集压测 sample_belt → 吐出**每次不同的荒诞 TypeError**(`type+float`、
  `range_iterator*range_iterator`、`BeltConfig not callable`)= 内存 corruption 铁证,非逻辑 bug。
  加 `OMP/OPENBLAS/MKL_NUM_THREADS=1` 后单进程 500 次 0 失败 → **numpy 多线程重入是诱因之一**。
- **修复**:`envs/belt_generator.py sample_belt` 热循环**去掉 `np.linalg.norm`**,改纯 Python 标量
  `sqrt(dot)`(更快且绕开崩溃代码路径)。16-worker forkserver build 崩溃率从旧码 ~100% 降到 ~25%。
  残余靠 `train_resilient.sh`(单线程 + resume 重试 12×)兜住:build 只发生一次,过了就稳训。
- **n135 训练已起**(`train_resilient.sh ppo_n135_... 8000000 135 --curriculum --traverse 300 700 --n-start 5
  --net-width 512 --exit-r-end 0 0 --ent-coef 0.05`):attempt1 崩 build、attempt2 进入训练并持续推进。
- **train_ppo.py 修**:EvalCallback `n_eval_episodes` 10→40(原 ±800 方差致 best_model 选噪声)。

## 2026-06-09 — 会话恢复:停 n135 训练 + 用最佳模型演示 + 规划两个新方向(待实现)

会话中断后恢复。用户在远程桌面(DISPLAY=:1)观看,要求**不开训练、不做破坏性操作**。
- **停掉**上次遗留的后台训练 `ppo_n135b_0609_1552`(n135/8M/curriculum,PID 31891/31897/31902)。
- **演示最佳模型** `models/ppo_traverse_n40_63pct.zip`:给 `Agent_tool/rollout_viewer.py` 加
  `--exit-r` 开关(对齐模型真实训练条件)。①简单版 n40/exit0:6 集 2 成功 2 撞(符合 63%)。
  ②困难版 n135/exit90-150:全撞/出界(n40 直线模型没训过这密度+偏离,符合预期)。build 报
  `105/135 fit (min_gap=55);30 parked`——n135 实际有效密度仅 ~105。
- **澄清模型谱系**:简单版(n40 直线穿越)已训通 63%(当前架构历史新高);困难版(n135+大偏离)
  从未训通,是下一座山。

**规划的两个新方向(用户拍板设计,本轮只 log、暂不实现):**
1. **加长陨石带**:改 `BeltConfig.belt_x_range`(如 600→1000m),目标 X 自动跟随末端;配套
   `n_asteroids` 按比例加保密度。**`max_steps` 要给得很宽松**(远超够用,不让它成为约束)——
   只有**严重超时**才截断,不能因步数不够误判失败。航程 curriculum 已就绪。
2. **新任务「飞到陨石带内部随机点」**(**新增 `goal_mode` 开关,保留 traverse**):
   - `reset()` 在 `interior_point` 模式下 X 在带内随机、yz 在截面内随机;**新逻辑**:目标点
     对所有小行星做 clearance 检查(避免落在石头里)。
   - **随机点需保证一定 X 方向深度**(不能太浅/紧贴入口):X 采样下界要离带子起点有最小 depth,
     强制飞船真正飞进带子。
   - **到达判定分档递进**(用户定的难度阶梯):①先「进入目标球即成功」(复用现有逻辑,最易) →
     ②加「进入且速度<阈值」→ ③阈值可配置/每集随机指定目标速度。
   - 架构天然支持:obs 已含本体系目标方向/距离,reward 已是任意目标点势能引导,网络/reward 基本不动。

**当前工作区(未 commit):** train_ppo(n_eval 40 + warm_belt_cache)、belt_generator(布局缓存绕
numpy corruption)、.gitignore(.belt_cache/)、rollout_viewer(+--exit-r)、日志。`models/ppo_traverse_n40_63pct.zip` 主力模型。

## 2026-06-09 — 整理:审查工作区 + 改正过时 41% 文档 + track 新主力模型

用户要求整理代码与 log(不新增功能)。审查结论:
- **未提交代码改动全部自洽**(train_ppo/belt_generator/.gitignore/rollout_viewer),无残留乌龙。
  台账与日志是追加式诚实记录(漏 ent / best_model bug 等乌龙已如实自纠),**不改写历史**。
- **唯一残留乌龙=过时的 41%** 散落 3 处,已改正(只改错信息,不加功能):
  - `README.md`:badge 41%→63%;Result 表 → 63%/oob0%/timeout0%/coll37%(标注 n40 直线)+ "Still open"
    块讲清大偏离&n135 仍未解;去掉"rewarded for doing it fast"(speed reward 早已删除);roadmap 同步。
  - `models/README.md`:新增 `ppo_traverse_n40_63pct.zip` 段(当前最佳, viewer 命令);v15 旧"修正"段
    改为"straight 已 63%、hard task(off-axis+n135)仍开放"(原写"当前架构从未训通/~2%"已过时)。
- **新主力模型 `models/ppo_traverse_n40_63pct.zip` 已 `git add`**(staged,未 commit,随 v15 进 git 快照区)。
- **保留的调试态(未动)**:`asteroid_belt_env.py:64 goal_radius=60 DIAG(temp)`(最终目标 35,属训练中间态);
  `eval_policy.py --n-asteroids` 默认 135(跑评估别忘了按模型传 40)。
- **`PROJECT_PLAN.md` 同步更新**:那段"从未训通/0–2%"→改为"直线已 63%、hard task(off-axis+n135)仍开放"+
  关键教训(net512/best_model bug);**两个新方向(加长带子、内部随机点)正式补进路线图**(待实现 checklist)。

## 2026-06-09 — 17 推进器:火焰可视化工具 + RCS 改对称布局(参数化)

用户想"看推进器喷口火焰"并质疑布局对称性。澄清+落地:
- **澄清:喷口角度固定,不可调**。当前 17 推进器每个都是固定方向单向力(`act.gear` 写死),6-DOF 靠
  "固定多喷口的位置/朝向组合"实现(符合真实 RCS 工程),**不是**矢量/万向节喷口。可调角度=新建模(动作空间
  涨到~51 维、映射变双线性非凸、探索更难),且真实模式还没开训,**用户决定先维持固定喷口**。
- **新增火焰可视化工具(`Agent_tool/`)**:`thruster_flames.py`(helper:按推力强度在每个喷口画 capsule 火焰,
  沿喷口反方向喷出,用 `mjv_connector` + viewer.user_scn 渲染,不碰物理);`thruster_flame_viewer.py`(动态:
  飞机做对称机动,火焰随推力 0→max→0 呼吸);`thruster_layout_viewer.py`(静态:锁住飞机、17 喷口全亮、
  按组着色 main橙/reverse红/rcs青 + 图例,`--cycle` 逐个点亮报名)。
- **RCS 改对称布局(仍 12 个,动作空间不变,无模型作废)**:旧布局是"最小喷口拼可控性"的产物——±Z 比 ±Y
  多一倍喷口(roll 力偶都是 ±Z)、前后喷口对角放(offset for roll)。**新布局**:8 竖直(±Z)在四角
  (±RCS_X, ±RCS_ZY, 0) 镜像对称(纯升力力矩天然抵消/前后差动=俯仰/左右差动=滚转)+ 4 水平(±Y)在前后中线
  (±RCS_X,0,0)(纯侧移/前后差动=偏航)。verify `wrench rank=6` 六轴正负全可达、env realistic reset/step OK。
- **几何参数化**:顶部 `MAIN_X/REV_X/RCS_X/RCS_ZY` 常量,改一个数所有相关喷口一起动,方便自调。
  用户最终调到 `RCS_X=6.0`(更贴机身)、`RCS_ZY=4.5`、`MAIN_X=-11`。
- **flame_viewer 机动表**同步对称版,新增纯 roll/pitch/yaw 旋转机动。
- 渲染命令:`DISPLAY=:1 conda run -n asteroid-belt-runner python Agent_tool/thruster_layout_viewer.py`;
  可控性自检:`conda run -n asteroid-belt-runner python envs/thruster_layout.py`。

## 2026-06-09 — 热启动复用 63% 模型起训 n80 + 落地两个新任务方向（加长带子/带内随机点）

会话目标：用户拍板"既然 63% 模型可复用，就热启动续练并监控；同时把之前说的新方向写好；确认 17 推进器是否已写"。
- **澄清"网络深度"问题**：当初突破是 **net 加宽**（256²→512²，仍 2 层），不是加深。"加层/更深"只是当时列的
  *候选杠杆*、从未触发（加宽就破局了），**不是遗留 bug**。目前瓶颈是冲撞局部最优/避障，不是容量，无需动深度。
  PROJECT_PLAN 记为"开放保险栏"（若再撞容量墙，给 train_ppo 加 `--net-depth`）。
- **17 推进器现状**：`thruster_layout.py`+`build_scene(realistic)`+env `dynamics` 开关+火焰可视化+对称 RCS
  **全部已建模&验证（wrench rank=6）**，唯独"在 17 维动作空间上训练"从未做（Phase 6/R10 仍开放）。
- **`train_ppo.py` 加 `--init-from <model.zip>` 热启动**：`PPO.load` donor 权重做新 run 种子（net_arch 随 donor=512，
  `--net-width` 忽略），ent_coef 取本 run CLI，timesteps 重置。`--resume`（崩溃恢复）优先于 `--init-from`，逻辑正确。
  仅 obs(160)+action(6) 不变时可复用（traverse/加长带/带内点 都满足；真实 17 推进器 6→17 不行，只能迁底层特征）。
- **起后台训练（子 agent 监控）**：`train_resilient.sh ppo_warmstart_n80_0609 4000000 80 --curriculum --n-start 40
  --net-width 512 --ent-coef 0.05 --exit-r-end 0 0 --init-from models/ppo_traverse_n40_63pct.zip`。
  warm-start 确认（512² checkpoint 8.38MB）；eval reward -1350→-458(350k)→curriculum 加密度后回落 ~-4000（n40→n80
  密度爬升期短 episode，预期低谷）；后半程满 n80 看能否突破。**训完子 agent 会测 final model.zip + best 两个、报 n80 成败分布。**
- **新方向①加长陨石带（实现完成）**：`--belt-len FAR` → `belt_x_range=(100,FAR)`；`--max-steps` 默认按带长自动放宽
  `max(2200, 2200×FAR/700)`（只严重超时才截断，不让步数误判）；密度 `--n-asteroids` 自己按比例加。
- **新方向②带内随机点（实现完成）**：env 加 `goal_mode`（默认 traverse 不变 / 新 `interior_point`）。
  `interior_point`：reset 先放石头→采 X 带内随机（`interior_min_depth=200` 保证最小深度不贴入口）、YZ 盘内随机
  （`interior_rho_frac=0.7`）、对所有 active 石头 `interior_clearance=30` 检查（不落石头里，200 次拒绝采样）。
  OOB 远界按模式分（traverse 跟 goal_x 防过冲 / interior 钉在 `belt_far_x` 带子出口）。
  到达分档：①进球即成功 → ②`arrival_speed=阈值`（进球且速度≤阈值，obs 不变）→ ③`arrival_speed_random=(lo,hi)`
  每集随机目标速度（**追加进 obs → 161**，让 agent 能遵守；故 tier3 与 160 维模型不兼容，属更难变体）。
- **工具链同步**：`eval_policy.py`/`rollout_viewer.py` 加 `--goal-mode/--belt-len/--arrival-speed`（默认全等于旧行为）；
  `--max-steps` 默认改自动放宽。**全部加法式改动**：默认路径 obs 仍 160、traverse 行为逐字节不变 → 不影响正在跑的训练/历史模型。
- **验证**：check_env 式 smoke（traverse obs160 / interior tier1-3 obs160-161 / 加长带 1100m 深度778.7≥200 / clearance35.6≥30）
  全过；4 文件 py_compile 全过。临时测试文件已删（未留根目录垃圾）。
- **未 commit**：train_ppo（--init-from/--belt-len/--goal-mode/--arrival-speed/auto max_steps）、asteroid_belt_env
  （goal_mode + 内部点采样 + 到达档位 + belt_far_x）、eval_policy/rollout_viewer（透传）、CLAUDE.md/PROJECT_PLAN/本 log。
  训练产物 `logs/ppo_warmstart_n80_0609`（git-ignored）。**等 n80 训练结果出来再决定 commit/留模型。**
