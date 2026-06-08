# 地基重建 TODO（2026-06-08 启动）

> 用户决定推倒占位地基、按"结构真实"重建小行星带，并一次性完成全部。
> **本文件是跨机器接续的唯一依据**（session 内的 TaskCreate 列表不会同步）。
> 换机器后：`git pull`，读本文件 + `Agent_log/CLAUDE_LOG.md`，从"当前进度"处接着干。

## 已锁定的决策（用户拍板）
1. **尺度** = 游戏化标度（紧凑、只让结构真实，不追求真实物理量级）。
2. **RL 动力学** = 简化 6 力先行（但把力调到合理量级、非瞬间机动），之后再迁 17 推进器。
3. **小行星运动** = 缓慢漂移 + 自转（动态 free-joint）。
4. **小行星外形** = 程序化凹凸"土豆"网格（非球！），视觉上像真实小行星。
5. **小行星带** = 结构真实（幂律尺寸分布、低相对速度、最小间距）但密度人为调高（真实密度会空无一物→任务无意义）。

## 架构变更要点
- **build-once / reset 重摆**：场景只编译一次；每回合用 `qpos`(位置+朝向) / `qvel`(漂移+自转) 重新摆放小行星，
  **不再每次 reset 重新编译**（旧实现每 reset 重建 model，慢；新实现快很多，且 mesh 资产稳定）。

## 任务清单与当前进度
- [x] **R1 小行星网格库**：`envs/asteroid_mesh.py`（icosphere 细分 + 径向高斯凸起/凹坑，纯 numpy 无需 trimesh）。
      已生成 `assets/asteroids/asteroid_0..11.obj`（各 320 面）。已验证 MuJoCo 能加载、按 scale 缩放、能碰撞。 ✅
- [ ] **R2 重写 `envs/belt_generator.py`**：一次建好 N 颗 free-joint 小行星 = 随机库 mesh + 每颗独立 mesh 资产
      (scale = 幂律尺寸 × 随机各轴 aspect) + 随机初始朝向；碰撞掩码(只撞飞船)；飞船碰撞代理胶囊。**不在 reset 重编译**。
      → ⚠️ **当前 belt_generator.py 还是旧的球体版**，尚未改。
- [ ] **R3 env reset 重摆**：每回合 numpy 采样小行星位置(带状区+最小间距+远离出生点)+随机朝向写 qpos，
      漂移/自转写 qvel；飞船状态复位；`mj_forward`。删掉旧的"每 reset 重建 model"。
- [ ] **R4 obs 加相对速度**：K 最近邻每颗 = 体坐标系 相对位置(3)+相对速度(3)+半径(1)+表面距(1)=8；obs_dim=16+8K。
- [ ] **R5 力调合理量级**：env 加 `max_force/max_torque`，用 `act_scale/act_bias` 把 action∈[-1,1] 映射到合理力/力矩
      (简化模式)，与 XML 巨 ctrlrange 解耦；真实模式保持 17 推进器 [0,max] 单向映射。
- [ ] **R6 重调奖励**：保留 potential + proximity(d_safe) + spin + collision/success/oob，按新尺度/漂移调参。
- [ ] **R7 更新工具**：`Agent_tool/{check_env(双模式), preview_belt(改用 env 显示漂移), eval_policy, rollout_viewer}`。
- [ ] **R8 验证 + 训练简化控制器**：check_env 双模式过 → 冒烟训练 → 课程 PPO 正式训练 → `eval_policy` 看成功率。
- [ ] **R9 键盘飞控（Phase 5）**：`manual_controller/main.py` 加 力模式(6)/推进器模式(17)，与现有运动学试玩模式切换。
- [ ] **R10 真实 17 推进器训练（Phase 6）+ 文档**：`--dynamics realistic` 训练并与简化对比；
      更新 `CLAUDE.md`/`PROJECT_PLAN.md`/`CLAUDE_LOG.md`；补 `requirements-rl.txt`。

## 当前进度（断点）
**R1 完成，R2 起未动。** 仓库处于"网格库已就绪、但 belt_generator/env 仍是旧球体版"的中间态。
下一步从 **R2 重写 belt_generator.py** 开始。

## 环境与命令
- conda env：`space-robotics-project`（MuJoCo 3.3.7 / Gymnasium 0.29.1 / SB3 2.3.0 / torch 2.2.1）。
- 重新生成网格：`conda run -n space-robotics-project python envs/asteroid_mesh.py`
- 训练/评估命令见 `PROJECT_PLAN.md` 末尾速查。
