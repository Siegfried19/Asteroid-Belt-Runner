# Asteroid-Belt Runner — 总体实施计划 (Project Plan)

> 维护者：Claude Code。每完成一个勾选项，在 `Agent_log/CLAUDE_LOG.md` 追加一条记录。
> 本文件是"做什么 / 按什么顺序做"的活路线图；具体设计细节可另开文档，这里留指针。

## 目标
控制《星际公民》F8C Lightning 穿越小行星带。先用**简化动力学 + 强化学习**跑通"能穿越"，
再升级到**真实 17 推进器动力学**重新训练。手动速度控制 (`main.py`) 保留为"试玩"入口，
不参与 RL（RL 用力/力矩或推力作为动作）。

## 技术栈（已确认可用，无需安装）
- conda env: **`asteroid-belt-runner`**（Python 3.10）
- MuJoCo 3.3.7 · Gymnasium 0.29.1 · Stable-Baselines3 2.3.0 (PPO) · PyTorch 2.2.1+cu121 · NumPy 1.26
- 注意：本 env 的 numpy 是 1.26（SB3/gym 要求 <2），与根目录 `requirements.txt` 里写的 2.3.3 不一致；
  RL 代码以本 env 为准，`main.py` 试玩在两者下都能跑。

## 目录约定（借鉴 Human_exo_interaction）
- `envs/` — Gymnasium 环境与场景生成器（RL 用）
- `train/` — 训练 / 评估脚本
- `Agent_tool/` — Claude 用的一次性小工具（冒烟测试、可视化、sanity check），git 跟踪
- `Agent_log/` — Claude 工作流水账（`CLAUDE_LOG.md`），git 跟踪并 push
- `TODO_list/` — 计划文档（本文件）
- `logs/` — 训练输出 / checkpoint / tensorboard（git 忽略）

---

## 阶段路线图

### Phase 0 — 脚手架 ✅ 完成
- [x] 确认 conda env 与 RL 依赖
- [x] 建立 `Agent_tool/ Agent_log/ TODO_list/ logs/` 目录与约定
- [x] `.gitignore` 区分"训练输出(忽略)"与"Agent_log(跟踪)"
- [x] 更新根目录 `CLAUDE.md` 记录新结构

### Phase 1 — 小行星带环境（Roadmap #1）✅ 完成
- [x] `envs/belt_generator.py`：程序化生成 N 颗小行星（带状区域内随机位置/半径/可选漂移，seed 可控）
- [x] 用 MjSpec 在运行时把飞船 + 小行星带组装成场景（避免手写巨型 XML）
- [x] 给飞船加碰撞代理（胶囊），并用 contype/conaffinity 掩码让小行星只与飞船碰撞、不互碰、不碰坐标轴
- [x] `Agent_tool/preview_belt.py`：viewer 里目视检查（需显示器，待用户本地跑）

### Phase 2 — 简化动力学 RL 环境（Roadmap #2）✅ 完成
- [x] `envs/asteroid_belt_env.py`：`gymnasium.Env`（obs 56 维 / action 6 维，通过 env_checker）
  - 动作：6D 力/力矩（复用 XML 里的 Fx Fy Fz Mx My Mz 执行器）
  - 观测：v_body(3)+ω_body(3)+前向(3)+上向(3)+目标方向(3)+目标距离(1)+最近K=8小行星×(相对位置3+半径1+表面距1)
  - 奖励：+向 +X 推进 − 控制 − 时间，碰撞 −100 终止 / 到达 +200 终止 / 越界 −50 终止 / 超时截断
  - reset：randomize_belt 时换 seed 重建带
- [x] `Agent_tool/check_env.py`：env_checker 通过 + 20 集随机动作 rollout（结果合理：无意外成功）

### Phase 3 — 训练简化控制器（Roadmap #2 续）✅ 达成（地基重建后）
- [x] `train/train_ppo.py`：SB3 PPO + SubprocVecEnv + checkpoint/best + tensorboard
- [x] `Agent_tool/rollout_viewer.py`：加载 checkpoint 在 viewer 里回放策略
- [x] 冒烟训练跑通（~5000 fps，PPO 更新正常，模型保存）
- [x] **正式训练达成**：地基重建后 `ppo_rebuild_v4` 课程 PPO 在满 60 密度带 **SUCCESS 100% / 0 碰撞**
      （mean_return 544±25；详见 `TODO_list/REBUILD_TODO.md` R8 与 `Agent_log/CLAUDE_LOG.md`）。

### Phase 4 — 真实动力学（Roadmap #3）✅ 建模完成
- [x] `envs/thruster_layout.py`：17 推进器（2 主 +X / 3 反向 −X / 12 RCS），每个 = site + site-actuator
      (gear=推力方向, ctrlrange=[0,max])。**已验证 6-DOF 可控**（wrench 矩阵 rank=6，每轴 ±均可达）。
- [x] `build_scene(dynamics="realistic")` 加 17 推进器；env 的 `dynamics` 开关切换简化(6)/真实(17)，
      只命令推进器执行器（6 个虚拟执行器留 0，因 MjSpec 3.3 不能删 actuator）。env_checker 双模式通过。
- [x] 推力上限按旧 6-DOF range 标定：主 5e6×2 / 反向 1.6e6×3 / RCS 1e6×12。
- [ ] （待 Phase 6）真实动力学下 reward 是否需重新调（姿态控制更难）

### Reward 调试记录（训练成功率）
- **v1**（x-progress reward）：success **4%**，oob 79%。"猛冲 +X 刷分后侧飞出界"。
- **v2**（potential reward = 到目标点距离减小量 + 朝向）：success **7%**，oob 降到 21% 但 collision 飙到 72%。
  修好了侧飞，但变成直冲撞石头（无提前避障激励）。
- **v3**（+proximity 惩罚 d_safe=12 / +spin 惩罚 / collision 100→300 / 课程 12→60）：训练中，待评估。
- 课程学习已实现：`env.set_n_asteroids()` + `train CurriculumCallback`（`--curriculum --n-start N`）。

### 2026-06-09 现状（直线穿越已破 63%；hard task 仍开放）
- 历史脉络：早期 100%/41% 都是**"直线穿越"任务**（无 off-axis exit）的成绩。commit `654bdb3` 引入
  **随机偏离出口 + speed reward** 后曾一度卡死，6 轮训练 0–2%（**冲撞局部最优**：progress 奖励沿途刷分
  把策略推向"全速冲撞"，比"不动"还差）。
- **突破（当前架构历史新高）**：去 speed reward + closing 逼近惩罚做避障主力 + **net512 + 航程 curriculum**
  后，**满航程 700m / n40 / 直线 = SUCCESS 63%、oob 0% / timeout 0%、仅剩碰撞 37%**。主力模型
  `models/ppo_traverse_n40_63pct.zip`。航程"恐惧倒退"oob 被 curriculum 治好。
- **关键教训**：①真正瓶颈一度是**网络容量**（net512 才学得动），别死磕 reward；②`EvalCallback n_eval_episodes`
  方差大→`best_model` 选噪声，**永远同时测 final `model.zip`**（已修 n_eval 10→40）。
- **hard task 仍开放**：①大偏离 off-axis exit（90-150）、②高密度 n135——这两档尚未训通，仍是冲撞局部最优。
- **可复用基础设施**：curriculum 密度+偏离+航程三维同步爬（`--curriculum --exit-r-end --traverse --ramp-frac
  --ent-coef --net-width`）；`eval_policy --exit-r/--n-asteroids` 指定难度评估；`rollout_viewer --exit-r` 可视；
  `belt_generator` 布局磁盘缓存（绕高密度 numpy corruption）；`render_scene.py` 离屏出图；场景 600m×180m。
- **下一步二选一**：①降 n40 碰撞 或 ②推 curriculum 加难（密度 40→80→135 / `goal_radius` 60→35 / off-axis 0→150）。

### 2026-06-09 续：n80 失败 + 发现"63% 是虚高、没真避障" → weaving curriculum（代码已就绪，待训练）
- **n80 热启动训练失败**：满 n80=0% success、final 模型 **100% 出界**（退化成"逃出界"避障）。根因=reward 平衡
  （`oob_penalty 200 < collision 600`，逃跑比硬穿便宜）+ 密度跳太陡，**非容量问题**。
- **关键诊断（`Agent_tool/diag_weaving.py`）**：63% 模型**没有真正避障**——成功集横向偏移仅 ~30m（带半径 180），
  且**正前方放一颗石头 → 100% 直撞、0% 能绕**（左右对称瘫痪）。**63% 本质=随机带子通常在轴附近留了条 ~30m 能钻的缝**，
  不是稳健避障；n80 把缝堵上就崩。
- **修复方向（用户拍板）**：①**速度自学**——不硬截 qvel、全局软速度上限默认关，靠 `w_closing`（只罚"朝石头冲快"，
  空旷不罚）+ 碰撞后果让它自己学会该慢则慢；②**单障碍 weaving curriculum**——`n_blockers`/`blocker_jitter` 强制把
  石头摆在 start→goal 路径上（1→K 颗、jitter 大→小=偏置→正中），逼它先学会绕一颗再加难。已实现并冒烟验证
  （blocker 准确落在路径上；63% 模型对单正中 blocker 5/5 全撞=baseline）。`train_ppo --n-blockers/--blocker-jitter
  (+-start)` 接入 CurriculumCallback（与密度/偏离/航程同步爬）。
- **下一步**：warm-start 63% 模型 + weaving curriculum（blocker 1→K、jitter 大→小、密度 稀→n40）训练，
  测 final model.zip（既测 blocker 场景也测普通 n40）。**待用户发令起训。**

### 新任务方向（2026-06-09 用户提出；**代码已实现，待训练**；详见 CLAUDE_LOG）
- [x] **加长陨石带（实现）**：`train_ppo.py --belt-len FAR` 设 `belt_x_range=(100,FAR)`，`--max-steps` 默认
      **按带长自动放宽**（`2200×FAR/700`，下限 2200，只严重超时才截断）。`--n-asteroids` 自己按比例加保密度。
      `eval_policy/rollout_viewer` 同加 `--belt-len`。 → [ ] **待正式训练**（拉长带子重训/续训）。
- [x] **「飞到带内随机点」(实现)**：env 新增 `goal_mode` 开关（默认 `traverse` 不变，新 `interior_point`）。
      `interior_point`：`reset()` 先放小行星，再采 X 带内随机（`interior_min_depth` 保证最小深度不贴入口）、
      YZ 截面盘内随机，对所有 active 小行星做 `interior_clearance` 检查（不落石头里）。到达分档：①进球即成功
      → ②`arrival_speed=阈值`（进球且速度≤阈值）→ ③`arrival_speed_random=(lo,hi)` 每集随机目标速度（**追加进 obs**，
      obs 161）。`train_ppo/eval_policy/rollout_viewer` 加 `--goal-mode/--arrival-speed`。 → [ ] **待正式训练**。
- **热启动复用**：`train_ppo.py --init-from <model.zip>` 用已训模型权重做种子起新 run（net_arch 随 donor）。
      只要 obs(160)+action(6) 不变（traverse/加长带/带内点 都满足），63% 模型可直接续练；换真实 17 推进器不行。
- **网络深度（开放选项，未触发）**：当前 net 恒 2 层（`--net-width` 只控宽）。突破靠 256²→512² **加宽**；
      "加层/更深"从未试过，是若再撞容量天花板时的保险栏（需给 train_ppo 加 `--net-depth`）。

### Phase 5 — 两套动力学的键盘飞控（Roadmap #4）
- [ ] 扩展 `manual_controller.py`：在直接速度控制之外，增加力/推力模式，可切换
- [ ] `main.py` 增加模式选择

### Phase 6 — 真实动力学训练（Roadmap #5）
- [ ] 在 17 推进器动作空间上重训 PPO（`--dynamics realistic`），对比简化版表现

---

## 训练 / 评估命令速查
```bash
# 训练（简化）
conda run -n asteroid-belt-runner python train/train_ppo.py --timesteps 2000000 --run-name ppo_simplified_v2
# 热启动续练（从 63% 模型加难，net 随 donor=512）
conda run -n asteroid-belt-runner python train/train_ppo.py --init-from models/ppo_traverse_n40_63pct.zip \
  --curriculum --n-asteroids 80 --n-start 40 --net-width 512 --ent-coef 0.05 --exit-r-end 0 0 --timesteps 4_000_000
# 加长陨石带（带长 1100m，max_steps 自动放宽，密度自己加）
conda run -n asteroid-belt-runner python train/train_ppo.py --belt-len 1100 --n-asteroids 120 --timesteps 4_000_000
# 带内随机点任务（tier2：进球且速度≤12）
conda run -n asteroid-belt-runner python train/train_ppo.py --goal-mode interior_point --arrival-speed 12 \
  --init-from models/ppo_traverse_n40_63pct.zip --curriculum --n-asteroids 40 --n-start 20 --net-width 512 --ent-coef 0.05
# 训练（真实 17 推进器）
conda run -n asteroid-belt-runner python train/train_ppo.py --dynamics realistic --timesteps 3000000 --run-name ppo_realistic_v1
# 评估成功率
conda run -n asteroid-belt-runner python Agent_tool/eval_policy.py --model logs/<run>/best/best_model.zip --episodes 100
# 回放（需显示器）
conda run -n asteroid-belt-runner python Agent_tool/rollout_viewer.py --model logs/<run>/best/best_model.zip
```

---

## 待用户裁定的开放问题（先用默认值推进，可随时改）
1. **小行星是静止还是漂移？** 默认：Phase 1 静止 + 轻微随机漂移开关，先静止训练。
2. **观测里障碍物表示**：默认 K-最近邻相对坐标（K=8），后续可换成射线/lidar。
3. **RL 算法**：默认 PPO（连续动作、稳健）。若想要 SAC/DynSyn 再说。
4. **目标定义**：默认沿 +X 穿过带、到达带另一侧的平面即成功。
