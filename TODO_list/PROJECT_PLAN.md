# Asteroid-Belt Runner — 总体实施计划 (Project Plan)

> 维护者：Claude Code。每完成一个勾选项，在 `Agent_log/CLAUDE_LOG.md` 追加一条记录。
> 本文件是"做什么 / 按什么顺序做"的活路线图；具体设计细节可另开文档，这里留指针。

## 目标
控制《星际公民》F8C Lightning 穿越小行星带。先用**简化动力学 + 强化学习**跑通"能穿越"，
再升级到**真实 17 推进器动力学**重新训练。手动速度控制 (`main.py`) 保留为"试玩"入口，
不参与 RL（RL 用力/力矩或推力作为动作）。

## 技术栈（已确认可用，无需安装）
- conda env: **`space-robotics-project`**（Python 3.10）
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

### Phase 3 — 训练简化控制器（Roadmap #2 续）🚧 进行中
- [x] `train/train_ppo.py`：SB3 PPO + SubprocVecEnv + checkpoint/best + tensorboard
- [x] `Agent_tool/rollout_viewer.py`：加载 checkpoint 在 viewer 里回放策略
- [x] 冒烟训练跑通（~5000 fps，PPO 更新正常，模型保存）
- [ ] **跑一次正式训练**（如 1M~3M 步），确认 reward 上升、学到避障/穿越 ← 下一步

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

### Phase 5 — 两套动力学的键盘飞控（Roadmap #4）
- [ ] 扩展 `manual_controller.py`：在直接速度控制之外，增加力/推力模式，可切换
- [ ] `main.py` 增加模式选择

### Phase 6 — 真实动力学训练（Roadmap #5）
- [ ] 在 17 推进器动作空间上重训 PPO（`--dynamics realistic`），对比简化版表现

---

## 训练 / 评估命令速查
```bash
# 训练（简化）
conda run -n space-robotics-project python train/train_ppo.py --timesteps 2000000 --run-name ppo_simplified_v2
# 训练（真实 17 推进器）
conda run -n space-robotics-project python train/train_ppo.py --dynamics realistic --timesteps 3000000 --run-name ppo_realistic_v1
# 评估成功率
conda run -n space-robotics-project python Agent_tool/eval_policy.py --model logs/<run>/best/best_model.zip --episodes 100
# 回放（需显示器）
conda run -n space-robotics-project python Agent_tool/rollout_viewer.py --model logs/<run>/best/best_model.zip
```

---

## 待用户裁定的开放问题（先用默认值推进，可随时改）
1. **小行星是静止还是漂移？** 默认：Phase 1 静止 + 轻微随机漂移开关，先静止训练。
2. **观测里障碍物表示**：默认 K-最近邻相对坐标（K=8），后续可换成射线/lidar。
3. **RL 算法**：默认 PPO（连续动作、稳健）。若想要 SAC/DynSyn 再说。
4. **目标定义**：默认沿 +X 穿过带、到达带另一侧的平面即成功。
