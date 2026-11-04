# W1 可行性初查记录

日期：2026-11-05。当前机器的盘点与公开资料初筛已完成；G1 MuJoCo 运行尚未完成。

## 1. 机器和软件环境

| 项目 | 实测 | 结论 |
|---|---|---|
| GPU | GeForce RTX 3060，6 GB 显存 | 与方案假定 RTX 4070 12 GB 不同 |
| NVIDIA 驱动 / CUDA | 驱动可见；系统 CUDA Toolkit 13.0 | 需与项目运行环境匹配 |
| Python / PyTorch | Python 3.13.5；PyTorch 2.12.0+cpu | `torch.cuda.is_available() == False`，当前 PyTorch 无法用 GPU |
| MuJoCo | 未安装 | 未加载 G1 或运行仿真 |
| Isaac Lab | 未安装 | 未验证训练环境 |

**机器检查部分已完成；G1 静态仿真验收仍待办。** 当前可继续写文档、定义 DSL / EMB 和开发 CPU 指标脚本。GPU 策略与 G1 闭环应在目标硬件（4070 12 GB）环境验收。

## 2. G1 仿真资产

优先验证 Unitree 官方 [unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco) 的 G1 29 DoF 场景。官方文档列出 C++ 模拟器支持 G1；Python 示例列出的机器人范围不含 G1，因此首测走 C++ 场景。加载后仍需核对 G1 具体修订、关节名称 / 顺序、限位、惯量和 PD 参数，避免混用资产版本。

## 3. 追踪 checkpoint 候选初筛

| 候选 | 初筛判断 | 决定前需核实 |
|---|---|---|
| [HoloMotion](https://github.com/HorizonRobotics/HoloMotion) v2 motion-tracking | **优先做接口验证**。官方说明列出 G1 29-DOF 和 MuJoCo 回放；v2.0.0 仍处于发布准备阶段，质量仍在迭代 | checkpoint 下载权限、版本与许可、12 GB 显存适配、动作 / 观测顺序、PD 参数，先原样回放再编辑 |
| [PHC](https://github.com/kexul/PerpetualHumanoidControl) | 方法参考，不是可直接用于 G1 的权重。官方实现基于 SMPL humanoid / Isaac Gym，坐标和机器人构型不同 | 跨形态重定向成本、checkpoint 与数据许可 |
| [OmniH2O](https://github.com/LeCAR-Lab/human2humanoid) | 公开示例主要面向 H1；代码仓库标记 CC BY-NC 4.0 | G1 权重兼容与非商业许可限制 |
| [NVIDIA GR00T / GEAR-SONIC](https://github.com/NVIDIA/Isaac-GR00T) | 有 G1 whole-body 路线，但官方给出的推理建议为 16 GB+，超过方案的 12 GB | 不作为 MVS 首选，除非找到并验证较低显存路径 |

**暂定排序：**先验证 HoloMotion G1 checkpoint；无法取得或接口不符时，继续寻找轻量 G1 MuJoCo tracker。PHC 作为训练和评测方法参考。以上属于公开资料初筛，未下载权重或实测兼容。

## 4. 在线延迟计时协议

所有事件记录单调墙钟和仿真时间；每个请求附唯一 request ID、EMB 版本号和结果状态：

`t0 received` → `t1 coarse_response` → `t2 dsl_valid` → `t3 candidate_ready` → `t4 commit` → `t5 first_effective_state` → `t6 constraints_satisfied` → `t7 displayed`

- **首次粗响应：**`t5 - t0`。只统计快速代理明确支持的命令，并记录其与最终请求的偏差。
- **完整请求延迟：**`t6 - t0`。作为主要编辑延迟，同时报告成功样本 P50 / P95、请求总数、拒绝率、超时率和截止期违约率。
- **画面延迟：**`t7 - t0`。如果无法获得实际屏幕时间戳，报告渲染状态代理值并明确标注。
- **模块时间：**分别测 DSL、生成、排队、门控、控制周期和展示等待；端到端分位数直接从完整请求计算，不把各模块 P50 / P95 相加。
- 冷启动和预热后数据分开。超时 / 拒绝计作任务失败；仿真时间不能替代墙钟。

### 待决定的目标冲突

计划同时要求“仅编辑未来 200 ms 之后的参考”与“输入到动作可见响应 P50 <150 ms”。若 200 ms 是严格保护前瞻，P50 目标无法满足。此记录先固定测量定义；两项目标仍保留，保护策略或响应目标的取舍待项目负责人决定。不能把候选生成耗时冒充端到端响应。
