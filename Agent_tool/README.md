# Agent_tool

Claude Code 用的一次性 / 辅助小工具：冒烟测试、环境 sanity check、可视化预览、
策略回放等。区别于 `envs/` 与 `train/`（项目主代码）—— 这里放的是"为了验证/调试
而临时写的脚本"，但仍 git 跟踪，方便复用与跨会话接续。

每个脚本顶部用 docstring 说明用途与运行方式。默认从仓库根目录运行：
```bash
conda run -n space-robotics-project python Agent_tool/<script>.py
```
