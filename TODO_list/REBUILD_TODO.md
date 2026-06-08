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
- [x] **R2 重写 `envs/belt_generator.py`**：一次建好 N 颗 free-joint 小行星 = 随机库 mesh + 每颗独立 mesh 资产
      (scale = 幂律尺寸 × 随机各轴 aspect) + 随机初始朝向；碰撞掩码(只撞飞船)；飞船碰撞代理胶囊；最小间距防重叠。
      **build-once 不在 reset 重编译**。返回 `list[Asteroid]`(body/geom/joint 名 + r_eff 保守外包球)。✅
- [x] **R3 env reset 重摆**：reset 写 qpos(位置+随机朝向) / qvel(慢漂移+自转)，飞船复位，`mj_forward`，不重编译。
      课程密度用 `n_active`(激活子集)+把其余陨石**停泊**到界外 x=5000 实现(模型固定大小、密度可变)。✅
- [x] **R4 obs 改雷达式**（取代旧 K-最近邻）：**体坐标系全向球面雷达**。已实现 obs_dim=160(12×6格)。✅
      - 方向格 = 方位角 N_az × 俯仰角 N_el（默认 **12×6=72 格**，可调）；随飞船姿态转动。
      - 每格 **2 通道**：① 该方向最近障碍**表面距**（实际存 1/(1+surf) 或归一化，量纲稳定）
        ② **径向接近速度**（closing velocity，>0 表示在靠近）。
      - **无遮挡**：解析分桶——遍历每颗陨石按球心方向落格，**近的不挡远的**；
        **同格取最近**那颗（最危险）。陨石按**保守外包球**处理：半径 = `scale × 网格最大外接半径 r_eff`，
        雷达表面距 = `‖rel‖ − r_eff`（偏保守、早报警、安全）。
      - **实现**：纯 numpy 解析（O(格数×陨石数)，不用 mj_ray，训练不掉速）；输出格式与未来射线版一致，可无痛替换。
      - 其余 obs 不变：体系 v(3)+ω(3)+fwd(3)+up(3)+goal_dir(3)+goal_dist(1)=16。
      - **obs_dim = 16 + 2×N_az×N_el**（默认 16+144 = **160**）。
- [x] **R5 力调合理量级**：按**官方 F8C 数据**标定(见 [[f8c-performance-specs]])。简化 6 力用 sign-asymmetric
      映射(action=0→0力)：前推 +X=mass×103.5(10.55G)、反推 −X=mass×36.3(3.70G)、侧移 mass×40；
      力矩按"~2s 达官方最大转速"标定(roll140/pitch38/yaw35 °/s)，惯量经 body_iquat 转回 body 系取对角。
      XML 6 虚拟执行器 ctrlrange 拓宽为不夹断包络。实测加速度/转速全部精确命中。真实 17 推进器保持 [0,max]。✅
- [x] **R6 重调奖励**：potential + heading + proximity(d_safe=12) + spin + **G-load 惩罚**(超 g_safe=6G 扣分，
      保飞行员命) + ctrl/time + collision300/success200/oob100。结构定稿；**权重数值留 R8 按失败模式微调**。✅
- [x] **R7 更新工具**：preview_belt 改用 env 显示漂移+自转；check_env 双模式过；rollout_viewer/eval_policy 兼容
      (randomize_belt 保留为忽略参数，随机带由 reset 重摆实现)。✅
- [x] **R8 验证 + 训练简化控制器**：课程 PPO(16 envs / 8→60 / 3M)。**`ppo_rebuild_v4` best：SUCCESS 100%、
      0 碰撞 / 0 出界 / 0 超时**(满 60 密度 100 集，mean_return 544±25)——简化控制器完美穿越！三处关键修复：
      ①reward 重平衡(proximity 0.4→0.05 软提示 / w_dist→2.0 / g_safe→8)治好"侧漂出界"；
      ②碰撞改几何检测(`_capsule_collision`，禁用飞船↔陨石物理接触)；
      ③**崩溃根治 = 单线程钉死**(`OMP/MKL/BLAS_NUM_THREADS=1`+`torch.set_num_threads(1)`，消除线程超额订阅内存损坏)。
      主力模型 `logs/ppo_rebuild_v4/best/best_model.zip`。✅
- [ ] **R9 键盘飞控（Phase 5）**：`manual_controller/main.py` 加 力模式(6)/推进器模式(17)，与现有运动学试玩模式切换。
- [ ] **R10 真实 17 推进器训练（Phase 6）+ 文档**：`--dynamics realistic` 训练并与简化对比；
      更新 `CLAUDE.md`/`PROJECT_PLAN.md`/`CLAUDE_LOG.md`；补 `requirements-rl.txt`。

## 当前进度（断点）— 2026-06-08 大重设计后
**简化动力学控制器已学会穿越大带：v15 = SUCCESS 41% / 出界 0% / 碰撞 59%**（满密度 100 集）。
主力模型 = `models/ppo_v15_best_41pct.zip`（=v15 的 1.5M checkpoint，后段发散故手动挑峰值）。

**带最终定型**（见 `belt_generator.BeltConfig` 默认值）：n=40(带内~31)、min_gap=55(缝~58m)、半径120×长400、
goal x=540、盒子碰撞贴 STL(机翼算，~26m 宽)、oob_margin=25、廉价 reset(预算布局+绕X轴旋转)。

**崩溃已根治**（关键，换机器务必保留）：本环境 CPython `sre_compile` 偶发损坏，import 期发作。
train_ppo.py 已用 **forkserver + torch 移进 main() + _warm_import 重试**；长训用 `Agent_tool/train_resilient.sh`
(崩了自动 --resume)。**别改回 spawn(狂崩)或 fork(死锁)。** 详见 `CLAUDE_LOG.md` 顶部。

**改进方向（下一步可选）**：① 治后段发散(早停/降LR，让它稳在/超过 47%)；② 降碰撞(调 proximity 奖励)；
③ R9 键盘飞控；④ R10 真实 17 推进器训练(`--dynamics realistic`)。

## 环境与命令
- conda env：`asteroid-belt-runner`（MuJoCo 3.3.7 / Gymnasium 0.29.1 / SB3 2.3.0 / torch 2.2.1，numpy<2）。
  换机器照 `CLAUDE.md` Setup 段重建同名 env。
- 长训：`Agent_tool/train_resilient.sh ppo_<name> 3000000 40 --curriculum --n-start 8 --max-steps 3000`
- 评估：`conda run -n asteroid-belt-runner python Agent_tool/eval_policy.py --model models/ppo_v15_best_41pct.zip --episodes 100 --n-asteroids 40`
- 回放(需显示器)：`Agent_tool/watch.sh models/ppo_v15_best_41pct.zip 30 40`
- 重新生成网格：`conda run -n asteroid-belt-runner python envs/asteroid_mesh.py`
