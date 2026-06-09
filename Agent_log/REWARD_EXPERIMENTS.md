# Reward / RL 超参实验台账（自主迭代）

**目标**：先在 `exit_r=0 + n=80 + 放大场景(700×180)` 上训出高 success（≥60%），找到能破解
"冲撞局部最优"的 reward 曲线 + PPO 超参；**再阶梯加 off-axis exit**（0→30→60→90→150）找
"通过率 vs 偏离"边界。每轮只动一个/一组变量、便于归因；后续改动参考本表历史。

评估口径：`eval_policy --episodes 100 --n-asteroids 80 --exit-r <训练时偏离>`，best_model，deterministic。

## 关键机理（已确认）
- **冲撞局部最优**：progress/speed 奖励沿途稠密刷分→策略学"全速冲撞"，比"不动"(安全 timeout ≈ -66)还差，
  训练 eval reward 常死锁在 -297。机动性充足(fwd 103.5 / lat 40 m/s²)，非物理瓶颈。
- **两个钟摆极端**：避障太弱→冲撞(coll 高)；避障太强/堵间隙→磨蹭(timeout 高)。正解在中间。
- **d_safe 不能 > 间隙半宽**：间隙 55m、船 26m→可用半宽 ~14m。d_safe 若 >14 则穿任何间隙都挨罚→磨蹭。
- **speed reward 是冲撞元凶**：一恢复就把策略推回冲撞，必须 w_speed=0。

## 实验表
| ver | 关键改动(vs 前) | success | coll | oob | timeout | 结论 |
|-----|----------------|--------:|-----:|----:|--------:|------|
| v16 | 原reward, 放大+大偏离(135/90-150) | 0% | 84% | 16% | 0% | 冲撞;progress 太主导 |
| v17 | w_dist2→1, coll300→900, prox0.05→0.6, succ→400 | 1% | 87% | — | 0% | 仍冲撞 |
| v18 | 易curriculum n5→90/exit0→150, ent0.01, gr→35 | 1% | 67% | 32% | 0% | 仍冲撞 |
| baseline | 原版难度 n40/exit40-90/原reward | 2% | 63% | 35% | 0% | 早期可学性已丢 |
| stage1 | 我的reward, **exit0**, n80 | 0% | 68% | 32% | 0% | exit0 也冲撞→非偏离问题 |
| **reward2** | **去speed**, d_safe45, w_prox6二次, w_close0.3 | 0% | **0%** | 2% | 98% | ✅破冲撞 但 d_safe45堵间隙→磨蹭 |
| **reward3** | 恢复speed0.02, d_safe15, +d_close50宽预警 | 0% | 75% | 25% | 0% | speed 一恢复就冲撞 |
| reward4 | 去speed + d_safe15 + d_close50 | 0% | **0%** | 44% | 56% | 半破:不撞✓但**过度避障**→横逃出界(oob44)+磨蹭(to56);缺目标牵引 |
| reward5 | w_dist1→1.5 + coll900→1200 + w_heading→0.15 | 0% | 1% | 2% | 97% | oob止住但磨蹭(97%) |

**🔑 [reward5 后重大转向] 真因不是避障,是"恐惧退场":** 密度扫描——reward5 模型在稀疏场 n=15/30 **全 oob**;
轨迹诊断显示飞船**往 -X 倒退到 x=-50 出界(远离目标!)**、峰值速度仅 25。结论:`collision=1200` + 强 avoidance
把飞船**吓得不敢前进**——学到"远离一切最安全"→稀疏场倒退出场、密集场原地磨蹭,就是不肯穿越。
**对策:降 collision 恐惧 + 提 success 吸引,让飞船敢于尝试穿越。** 后续若仍退缩→可能要加"后退惩罚"或限制后界。

| reward6 | coll1200→600 + succ400→600 | 0% | 71% | 29% | 0% | 降collision→敢动(to97→0)但又冲撞(71);钟摆 |

**[reward6 后] collision 钟摆死结:** 高(1200)→恐惧磨蹭,低(600)→冲撞。靠 progress 牵引前进 + collision 防撞
**本质冲突**(w_dist 越高越需高 collision,但高 collision→恐惧)。**突破:把避障责任从 collision(恐惧式)
转给 closing penalty(梯度式)**——强化 closing 做独立避障主力,collision 保持低(不恐惧),progress 放心牵引。

| reward7 | w_closing0.3→0.8 + d_close50→60, coll留600 | 0% | 4%@n80 | 56–100% | 0% | 强closing压住撞,但全oob(稀疏100%);仍0 success |

**🔑🔑 [reward7 后 元层面诊断] reward 钟摆走不通——6 轮(reward2–7)从未出现 1 次 success。** collision/oob/timeout
三者打地鼠、压一个起一个,飞船**从没真正到达过**。强烈信号:**PPO 没探索到"成功到达"轨迹,无从强化** →
**探索/任务难度问题,非 reward 平衡**。→ 停钟摆,回**极简任务**验证 PPO 能否学会基本"飞到目标":
航程 700→300m、success 球 35→60、n=2、强探索 ent0.05。能 success→真 curriculum 逐步加难;仍 0→更深 bug(控制/obs)。

### 极简验证阶段(DIAG)
| ver | 配置 | success | coll | oob | timeout | 结论 |
|-----|------|--------:|-----:|----:|--------:|------|
| diag_min | 航程300/球60/n2/ent0.05/exit0 | **22%** | 0% | 78% | 0% | 🎉**突破!史上首次>0**;train reward 转正+128 |

**🎉 [突破] PPO 能学会到达——是难度问题、不是根本 bug。** 极简 success 22%、训练 reward 转正(+128)。
确认 pipeline+reward 正确;之前满难度 0% 是探索不到成功轨迹。78% oob = "精确到达控制"还弱。
**下一步:①查 oob 方向(冲过头/横偏)对症提精度;②从极简点用真 curriculum(航程→密度→球→偏离逐档)爬到目标难度。**

| diag_r8 | 修oob后界 + w_heading0.15→0.3 | 0% | 0% | 100% | 0% | ⚠️退步(22%→0)!同改2变量难归因,疑 heading0.3 过强乱了位置控制 |
| diag_r9 | 回退heading只留oob修复 | 0% | 0% | 100% | 0% | **oob修复才是元凶!** 紧后界370是隐式刹车,放宽到420→飞船冲过球→0% |

**[diag_r9 诊断] oob 紧后界=有用的隐式刹车,非 bug。** 真缺的是"到达减速"机制(主动刹车进球)。
**diag_min(oob370,heading0.15,ent0.05)=22% 是当前最佳基线。** → 回退 oob 后界 + 加"接近目标罚速度"reward。

| diag_r10 | 回退oob + 到达减速(w_arrive0.15) | 10% | 0% | 90% | 0% | 到达减速也退步(22→10)!diag_min仍最佳22% |

**[diag_r10 后] reward 微调到顶:** diag_min 22% 是当前 reward 结构局部最优,3 次改进(oob/heading/arrive)全退步。
78% oob(冲过头/横偏)是 **6-DOF 控制精度**问题,调 reward 权重突破不了。→ **转 PPO 超参**(台账决策树)。
ent0.05 高探索助找成功(0→22%)但限精度。

### PPO 超参阶段
| ver | 配置(reward=diag_min) | success | coll | oob | timeout | 结论 |
|-----|------|--------:|-----:|----:|--------:|------|
| ent02 | ent0.05→0.02 | 6% | 0% | 94% | 0% | 降ent退步(22→6)!ent0.05探索是22%关键,不能降 |

**[ent02 后] diag_min 22% 在 reward+ent 都到顶。** 上更根本的 setting:**VecNormalize(norm_reward) 奖励归一化**
——reward 尺度悬殊(progress 上千/avoidance 几十/coll 600)让 value function 难学,归一化可能稳定突破。
只 norm_reward(不 norm_obs)→eval 端不受影响。

| vecnorm | +VecNormalize(norm_reward) | 23% | 14% | 63% | 0% | 没突破(22→23噪声内)+引入collision14;归一化扰乱避障 |

**[vecnorm 后] reward/ent/归一化全试遍,稳卡 22–23%。** 78%oob是6-DOF控制精度。剩余明显杠杆=**网络容量+训练量**(未试)。
→ net 256²→512² + 训练 2M→4M;VecNormalize 改 --flag 默认关(它引入了 collision)。

### 网络/训练量阶段
| ver | 配置(diag_min reward) | success | coll | oob | timeout | 结论 |
|-----|------|--------:|-----:|----:|--------:|------|
| net512 | net256²→512² + 4M步 | **100%** | 0% | 0% | 0% | 🎉🎉**完美突破!return 1046±1.1完全收敛** |

**🎉🎉🎉 [突破] net512+4M = 极简任务 100% success!** 22%→100%(0撞0出界0超时)。**真正瓶颈是网络容量
(256²表达不了精确6-DOF控制)+训练量,不是 reward!** 前 20 轮 reward 调试其实在错误维度打转——
reward 结构(diag_min)早够用,缺的是网络。教训:**遇到学不会先怀疑容量/训练量,别只盯 reward。**
→ 用 net512 从极简 100% 逐步加难逼近最终目标(放大+大偏离)。

### 加难阶段(net512 + 4M + ent0.05 + diag_min reward,从极简100%出发)
| ver | 加难维度 | success | coll | oob | timeout | 结论 |
|-----|------|--------:|-----:|----:|--------:|------|
| n40 | 密度 n2→40(航程300/球60/exit0) | 72% | 28% | 0% | 0% | ✅net512能避障!加密度只掉到72% |
| n40_long | +航程300→700 | 0% | 15% | 85% | 0% | 航程断崖!300→700跳太大,长航程横偏累积出界(85%) |

**[n40_long] 航程是断崖维度.** net512 短航程精确,长航程误差累积出界。→ 训中间档 500m 验证渐进还是断崖。

| n40_500 | 航程700→500(中间档) | 0% | 0% | 90% | 10% | 航程断崖!400m也崩 |

**[n40_500 诊断] 又是"倒退逃跑"!** oob 全 x=-50(朝-X倒退出界、远离目标),peakspeed低。长航程+n40 下飞船恐惧前进、
倒退逃(reward5 同病复发)。net512极简能前进,难度一高就复发。→ **加倒退惩罚(x<0每米重罚)堵死逃跑后路**,逼前进。

| n40_500_retr | +倒退惩罚 w_retreat2.0(x<0每米罚2) | 训练中 | | | | 堵死倒退,逼前进穿越 |

## 当前 reward 公式（env 默认，reward4）
- progress: `w_dist=1.0 × Δ(到目标距离)`
- heading: `w_heading=0.05 × dot(机头, 目标方向)`；speed: **w_speed=0**(关闭)
- **closing penalty(宽预警)**: `surf<d_close=50 且 closing>0 → -w_closing=0.3 × closing × (1-surf/50)`
- **proximity penalty(窄+二次)**: `surf<d_safe=15 → -w_proximity=6 × ((15-surf)/15)²`
- spin `0.01`, gload `0.15`(>8G), ctrl `0.001`, time `0.03`
- 终局: collision **-900**, success **+400**, oob **-200**; goal_radius 35

## 下一步决策树（reward4 结果出来后）
- **若 success 起来(≥40%)**：✅找到正解→阶梯加偏离(exit-r-end 30→60→90→150)，每档训+评估记本表。
- **若又冲撞(coll 高)**：closing 预警不够→提高 w_closing 或 d_close；或降 progress(w_dist→0.6)。
- **若又磨蹭(timeout 高)**：前进动力不足→升 w_dist(→1.5) 或降 time_cost；检查 d_safe 是否仍堵。
- **若卡住不破**：转 PPO 超参——ent_coef↑(0.01→0.03 更多探索)、lr↓(3e-4→1e-4 更稳)、
  n_steps↑、gamma(0.995)、net_arch(256²→512² 或加层)、reward normalization(VecNormalize)。

## 🚩 当前状态(2026-06-09 停在切换机器)& 下次 TODO
**最大成果:`net512`(512²网络)+ 4M 步是关键突破——极简任务 100%、n40 短航程(300m) 72%。真正瓶颈是网络容量,
不是 reward!** 前 20 轮 reward 调试卡 22%,换大网络直接破。
**当前障碍:航程断崖**——航程 300→500 飞船恐惧倒退、success 0%。**已实现航程 curriculum**
(`env.set_traverse` + CurriculumCallback ramp goal 距离 + `--traverse START END`,冒烟验证 OK)。
**当前 env 默认值是调试态**:goal_radius 60(放宽,最终要收 35)、reward=diag_min(去 speed/closing 避障主力/
anti-retreat w_retreat2)、belt_x 700(build 全范围)、belt_yz 180、n135/exit_r 90-150(大偏离默认,训练时 `--exit-r-end` 控)。

**下次 TODO(从这继续):**
1. **训航程 curriculum**(验证能否解决断崖):
   `--curriculum --traverse 300 700 --n-asteroids 40 --n-start 5 --net-width 512 --timesteps 4_000_000 --exit-r-end 0 0`
   评估:`eval_policy --model logs/<run>/best/best_model.zip --n-asteroids 40 --exit-r 0 0`(env 默认 goal 740 满航程)。
2. 成功后逐档加难:密度 40→80→135、收球 goal_radius 60→35、加偏离 `--exit-r-end 0 0`→`30 60`→`90 150`。
3. 每档 net512+curriculum,台账记录。**最终目标:放大(700×180×135)+大偏离(90-150)+球35 高 success。**
- 已验证最佳模型:`logs/ppo_diag_net512`(极简 100%)、`logs/ppo_n40`(n40 短航程 72%)。

## 待探索变量清单
- **reward**: w_dist, w_proximity, d_safe, w_closing, d_close, collision_penalty, success_bonus,
  time_cost, w_heading, goal_radius；惩罚曲线形状(二次/反比/指数)。
- **PPO**: learning_rate, ent_coef, n_steps, batch_size, n_epochs, gamma, gae_lambda, clip_range,
  net_arch, VecNormalize(obs/reward 归一化), 总步数。
- **curriculum**: n_start, ramp_frac, exit-r-end 节奏。
