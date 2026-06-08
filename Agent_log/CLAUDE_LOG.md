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

## 2026-06-08 — ✅ 大带学会穿越(41%)+ 崩溃根治(forkserver)+ 真凶=flaky sre_compile

- **崩溃真凶最终定论**:不是内存/陨石/envs/reset——是这个 conda 环境的 **CPython `sre_compile`(正则编译)
  偶发损坏**,谁在 import 时编译复杂正则都可能挂(torch、torch._dynamo、matplotlib、gymnasium 都中过招),
  报各种怪异解包错(`too many values to unpack`)。**只在 import/启动期发作,跑起来就没事。**
- **根治三连**:①start_method=**forkserver**(干净服务进程 import 一次、worker 从它 fork,不重复踩、也不
  fork-after-CUDA 死锁);②torch/SB3 移进 `main()`(worker 完全不碰 torch);③`_warm_import` 重试预热
  torch._dynamo/matplotlib + wrapper 重启兜主进程启动偶发。**v15 forkserver 全程跑满 3M,只崩 2 次。**
  (踩坑记录:spawn→每个worker重复import狂崩;fork→fork-after-CUDA死锁hang;forkserver 才对。)
- **带最终定型**(为"可学"放宽):n=40(带内~31)、min_gap **55**(缝~58m,对 26m 盒子碰撞飞机有 ~32m 余量,
  缝≈2.2×飞机)、半径120×长400、盒子碰撞贴STL(机翼算)。之前 min_gap32(缝仅4m余量)= 飞机挤不过→0%。
- **v15 训练**(课程8→40,3M):**1.5M checkpoint 达峰 SUCCESS 41% / 出界 0% / 碰撞 59%**(100集)——
  学会从缝里穿、不再侧漂钻空子(对比旧"绕飞带"的假100%)。**后段发散**(2M→0%),典型 PPO 后段不稳;
  resume 重置了 EvalCallback best 记录,故手动挑 1.5M checkpoint 存为 `best/best_model.zip`。
- **主力模型**:`logs/ppo_rebuild_v15/best/best_model.zip`(=1.5M checkpoint,41%)。改进方向:早停/降LR 治后段
  发散、proximity reward 降碰撞率。看回放:`Agent_tool/watch.sh logs/ppo_rebuild_v15/best/best_model.zip 30 40`。

## 2026-06-08 — 任务改造(随机出口+求快) + Python 3.11 根治尝试(失败,保留 forkserver)

- **任务改造**(用户要加难度)：①**随机偏轴出口**——每回合目标改为点 `(goal_x, gy, gz)`,gy/gz 偏轴
  40–90m 随机;势能/obs/heading 全改指向这个 3D 出口;成功=进出口点 `goal_radius=25m` 球(不再"飞过平面就赢",
  逼飞船绕到指定出口、不能直冲)。②**求快**——加 `w_speed=0.02 ×(速度·目标方向)` 奖励 + time_cost 0.02→0.03。
  check_env 双模式过,出口随机生效。
- **Python 3.11 根治尝试 → 失败**:建 abr311(3.11+同栈)测 `import torch._dynamo`,第一次 0/20(运气),
  改名为 asteroid-belt-runner 后 4/20——**3.11 没修好**,而且 3.11 里它直接 **段错误(core dumped)**,
  说明根因是 **torch 2.2.1 的 `_dynamo` 导入在 C 层偶发段错误**(3.10 的 sre_compile ValueError 只是同一崩溃
  的 Python 层表象),**与 Python 版本无关**。根治只能换 torch(2.3+,有 numpy/sb3 兼容风险)。
- **决定**:env 切到 3.11(和 3.10 等价、略快,旧 3.10 已删,check_env 过)+ 补 imageio;**forkserver 仍是真正
  的崩溃解法**(已写进 CLAUDE.md 警告)。是否赌 torch 2.3 待用户定。

## 2026-06-08 — 保存收尾 / 跨机器提交（用户将换机器）

- 用户满意现状(看了回放,41% 穿越),要求保存一切以便换机器。
- **提交**(分支 `rebuild/asteroid-belt-rl`)：`10ef9be` 大带重设计+41%+forkserver 崩溃根治；
  `59baeec` 跟踪 v15 best 模型快照。
- **主力模型入 git**：拷到 `models/ppo_v15_best_41pct.zip`(=v15 1.5M checkpoint)。踩坑:`.gitignore`
  的 `*.zip` 把它也忽略了,加例外 `!models/*.zip`——但**`.gitignore 不支持行内注释**(我把注释写行内
  导致 pattern 失效),注释挪到单独一行才生效。+ `models/README.md` 说明模型/用法/改进方向。
- **接续文档**：更新 `REBUILD_TODO.md` 断点段(现状=41%模型、带定型参数、forkserver 警告、下一步)。
- **推送**：本环境无 git 凭据/gh,我推不了;用户自行 `git push -u origin rebuild/asteroid-belt-rl` 完成。
- 换机器接续:clone → checkout 该分支 → 照 CLAUDE.md 建 `asteroid-belt-runner` env → check_env → 读 REBUILD_TODO。

## 2026-06-08 — ⚠️ 崩溃真凶 = torch dynamo 正则 bug（不是内存/陨石/envs）

- **重大纠错**:之前 v5–v10 把频繁崩溃归因为"陨石多/envs多/reset重采样的内存损坏"——**全错**。
  v11 抓到完整 traceback:两次崩溃**都在 `torch._dynamo/skipfiles.py → re.compile → sre_compile`**,
  报 `ValueError: too many values to unpack (expected 0/92)`。这是 **torch 2.2.1 + Python 3.10 的
  TorchDynamo 正则编译 bug**(偶发),和陨石/MuJoCo/我的代码都无关。
- **为何一直没发现**:最初第一次训练就是这个错,我当时用 `TORCHDYNAMO_DISABLE=1` 绕过,早期
  v1–v4 都带它(v4 干净跑完 3M)。**但 `train_resilient.sh` 漏设了这个变量** → v5–v11 全裸奔踩坑。
  envs/陨石越多只是触发概率越高,造成"内存随规模损坏"的假象。
- **根治**:`train_ppo.py` 顶部 `os.environ.setdefault("TORCHDYNAMO_DISABLE","1")`(import torch 前)
  + `train_resilient.sh` export 同样变量。我们从不用 torch.compile,直接禁用 Dynamo 无副作用。
- **附带收获仍有效**:这轮排查顺手做的 min_gap32(保证可穿)、盒子碰撞(贴 STL/机翼算)、廉价 reset
  (build布局+旋转,1.1ms,50×快)、固定 body 数(72,69入带3停泊)都是真改进,保留。
- 启动 **v11**(72颗/缝≥30/盒子碰撞/dynamo修复/课程20→72/3M)验证全程无崩 + 能否学会穿越。

## 2026-06-08 — 缝隙/碰撞体诊断 + 廉价 reset 根治崩溃（v8/v9 → v10）

- **v5–v8 反复在大带上学不会**(eval 下滑、0% 成功)。诊断 v8 best(90颗大石头):0%成功、**70% 侧向出界**、
  30%碰撞,**平均只飞到 x=127**(刚进带口就垮)。根因不是训练,是**带本身堵死**:
- **缝隙诊断**(用户提出):最近邻表面间隙**中位仅 13m < 石头直径 27m**,**48% 的缝 < 飞机碰撞径 12m**——
  飞机物理上钻不过近一半的窄缝。根因:`min_gap` 只有 1.5m,大石头挤成团。→ **min_gap 1.5→32m**,
  保证每条缝 ≥32m > 飞机宽度(0% 穿不过)。
- **碰撞体改盒子**(用户问"碰撞球怎么算的、能否贴 STL"):原是 12m 胶囊、**忽略机翼**。STL 是扁宽板
  (24×26×6m),胶囊裹不住 → 改**贴紧的 OBB 盒子** half-extents(12.21,12.83,2.93)、中心 X+0.67,
  机翼现在算碰撞(`_box_collision` 盒-球检测)。飞机有效宽度 12→26m。配套放宽侧向边界 oob_margin 12→**25**
  (盖住飞机半宽 13m + 机动)。`add_ship_collision` 改 box geom;BeltConfig 用 `ship_box_half/ship_box_cx`。
- **廉价 reset 根治崩溃**:v9(min_gap32)崩溃暴增到**每 58k 一次**——min_gap 越大,`sample_belt` 每次 reset
  的拒绝采样越疯狂,正是崩溃主因。→ **建场时预算 16 套布局,reset 只挑一套 + 随机绕 X 轴旋转**(O(N))。
  reset 1.3ms(快 ~50×)、建场 1.5s。`_precompute_layouts` + 重写 `_place_asteroids`,不再每 reset 重采样。
- 注:min_gap 32 较紧,带内实际 ~75 颗(其余停泊界外)。启动 **v10** 验证(廉价 reset 后崩溃应大降)。

## 2026-06-08 — 飞机朝向修正 + 大带重设计（可视化迭代）→ v7

- **飞机朝向 bug**：用户目视发现机头朝绿轴(+Y),但 env 以 +X 为前向(推力/目标/reward 全按 +X)→ 飞船"侧着飞"。
  STL 机头沿 mesh -Z、翼展 Y、厚度 X。**踩坑**：先用 `euler` 修,被 MuJoCo 的 mesh 内部帧/`model.geom_quat`
  误导,渲出来还是错。改用**离屏渲染(EGL)+ 真实渲染矩阵 `data.geom_xmat`** 验证才靠谱:正确解 = 绕 Z 转 −90°
  (`quat="0.7071 0 0 -0.7071"`,pos 同步转正保持居中)。机头→+X(红)、翼→Y(绿)、顶→+Z(蓝)。纯视觉,不动物理。
- **可视化迭代工具**：`Agent_tool/watch.sh`(一条命令开 viewer);离屏渲 PNG + Read 自查朝向(俯视/iso)。
- **大带重设计**(用户驱动,反复 preview 调):原带太小→飞船绕外缘作弊(v4 100% 是钻空子)。
  最终:**半径 120 × 长 400(x 100→500)、目标 x=540、出界 rho>132**。密度按"平均间距 ≥ 飞机尺寸的倍数"定——
  飞机包络球直径 ≈ 30m / 碰撞横截面 12m;用户选"间距 43m ≈ 3.5× 碰撞径 / 1.5× 包络球"(从容可穿)→ **N=230**。
  关键数据:旧 600 颗间距才 20m < 飞机 30m = 根本钻不过(必 0%);故重设密度。
- **启动 v7**:`train_resilient.sh ppo_rebuild_v7 3000000 230 --curriculum --n-start 30 --max-steps 3000`
  (带变长→max_steps 1500→3000)。抗崩溃续跑。待评估。
- 注:v2/v4 是旧小带模型,已过时;v7 是新大带主力。`environment.xml` 加了 geom quat/pos(朝向);belt_generator
  尺寸/密度更新。check_env 双模式过(随机动作多 timeout,因带稀疏目标远,符合预期)。

## 2026-06-08 — 发现 v4 在"绕飞"作弊 + 加宽带 + 抗崩溃训练（v5→v6）

- **用户目视 v4 viewer 发现陨石带太小、飞船绕外缘飞过、没真穿越**。诊断坐实：带半径 45，但飞船穿越带
  x 区间时**平均侧偏 72**(远在带外)、最大 92——出界边界宽到 105 留了"绕行走廊"，100% 0 碰撞是钻空子。
- **修带几何(真·穿越)**：belt_yz_radius 45→**55**(填满横截面)、出界侧向余量 60→**12**(env 新增 `oob_yz_margin`，
  rho>67 即出界、堵死绕行)、n_asteroids 60→**90**(更密)。check_env 过(随机动作几乎绕不过、碰撞率高)。
- **保存渲染脚本** `Agent_tool/watch.sh`(封装 DISPLAY/conda，一条命令开 viewer)。用户远程桌面 DISPLAY=:1 可用，
  EGL 离屏渲染也可(4090)。
- **v5(新带)~450k 又段错误**——单线程没根治(v4 60颗干净跑完、v5 90颗 450k 崩；陨石多→episode 短→reset 频繁→
  `sample_belt` 高频→撞上罕见非确定性内存损坏)。判定：与其根治这个跨子系统的罕见崩溃，不如**让训练能从崩溃恢复**。
- **抗崩溃训练**：train_ppo.py 加 `--resume`(从 run 目录最新 ckpt 续，`reset_num_timesteps=False`，课程接续)；
  `Agent_tool/train_resilient.sh` 包装重试循环(崩了自动 --resume，最多 12 次，直到"saved final model")。resume 已验证。
- 删崩溃的 v5；启动 **v6**(`train_resilient.sh ppo_rebuild_v6 3000000 90 --curriculum --n-start 10`)。待评估新带真实成功率。
- v4(旧带 100%)保留——但属"绕飞带"，新主力将是 v6(真穿越带)。

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
