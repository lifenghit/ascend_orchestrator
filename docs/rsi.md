# RSI 与效率看板

## 一键演示

```bash
PYTHONPATH=src python3 -m ascend_orchestrator rsi-demo --reset
```

演示使用同一个混合任务依次运行三层 Campaign：

| 层级 | 模拟结果 | 说明 |
| --- | --- | --- |
| OPERATOR | `KERNEL_CANDIDATE_ACCEPTED` | 微基准时延从 10 ms 降到 8 ms |
| MODEL | `MODEL_CANDIDATE_ACCEPTED` | 固定模型签名下通过集成验证，时延降到 8.5 ms |
| E2E | `REJECTED_NO_IMPROVEMENT` | 同一局部候选使真实负载代理指标退化到 11 ms，端到端否决 |

这些数字只用于验证决策逻辑，不是 NPU 测量结果。

## Campaign 生命周期

```text
冻结通过正确性门禁的基线
  → 固定主指标、方向、协议和最小有效收益
  → 注册假设、父候选、动作类型和候选负载摘要
  → 绑定真实执行 EvidencePack
  → 检查正确性、协议、模型/负载签名和环境
  → 接受、拒绝、要求复测或标记测量无效
  → 更新最佳候选、轮次和无改善计数
  → checkpoint / 停止
```

每个 Campaign 只有一个 Leader。默认 Leader ID 为 `rsi-leader-01`，其他 Leader 不能追加候选、记录评估或停止 Campaign。

### Comparison Contract

必填字段：

- `primary_metric`：主指标名称。
- `direction`：`MAXIMIZE` 或 `MINIMIZE`。
- `correctness_gate`：必须为 `REQUIRED`。
- `min_improvement`：最小有效收益。
- `measurement_noise`：测量噪声阈值。
- `retest_on_noise`：差异落在噪声范围时是否要求复测。

基线 EvidencePack 的 `performance` 必须包含数值主指标和 `protocol_digest`。候选评估必须使用相同协议、基础 commit、模型/工作负载/部署签名、RL 契约版本和环境签名。候选代码补丁和候选构建产物允许变化。

### 预算与恢复

`max_rounds`、`max_no_improvement_rounds`、`max_wall_time_minutes` 均为必填正数。`max_candidates` 可选，默认等于最大轮数。

`RETEST_REQUIRED` 和 `MEASUREMENT_INVALID` 不消耗有效轮次，但候选数量仍受预算限制。`campaign_checkpoint` 将轮次、最佳指标、接受候选、无改善计数和状态保存到 SQLite。

## 分层动作空间

| 层级 | 允许动作 |
| --- | --- |
| OPERATOR | `KERNEL_ONLY` |
| MODEL | `OPERATOR_SET`、`GRAPH_ONLY`、`MEMORY_ONLY`、`PARALLEL_ONLY`、`RUNTIME_ONLY`、`HYBRID` |
| E2E | `KERNEL_ONLY`、`MODEL_ONLY`、`RUNTIME_ONLY`、`HYBRID` |

模型候选的 EvidencePack 必须包含 `integration_verified=true`。E2E 候选必须包含 `e2e_validated=true`，RL 闭环候选还必须包含 `closed_loop_valid=true`。缺少这些字段时，即使主指标更好也不会接受。

模型或 E2E Campaign 创建算子子 Campaign 时，父候选必须携带 `bottleneck_confirmed=true`，表示已有 profiling 证据。嵌套仅允许一个有独立预算的层级，子 Campaign 不会自行接受上层目标。

## RL 联合候选

训练和推理分别建立 `role=TRAINING`、`role=INFERENCE` 的模型 Campaign，并绑定各自基线和契约版本。只有独立评估已接受的候选能进入联合批次。

`campaign_joint_batch` 要求联合 Campaign 为 `level=E2E`、`role=RL_INTEGRATION`，并执行以下过滤：

- 拒绝 `PROPOSED`、测量无效或未通过独立初筛的候选。
- 拒绝训练、推理和联合基线之间契约版本不一致的组合。
- 通过 `max_combinations` 限制组合数量，不无预算枚举完整笛卡尔积。

## 限时实验窗口

`measurement_window_create` 将多个 READY Step 绑定到同一开始时间和隔离范围，支持 `NODE_SET_QUIET` 与 `GLOBAL_QUIET`。Step 自身必须声明相同隔离等级。

`measurement_window_close` 要求窗口内没有运行 Step，取消尚未使用的预留，并写入 Campaign checkpoint。普通调度在窗口外恢复工作保持。

## 阶段四调度优化

- 同类型、同资源范围的成功执行形成历史时长分布，回填使用 P90，而不是 Agent 主观估计。
- 单卡分配采用 best-fit 装箱，尽量避免把大任务所需节点拆碎。
- `interference_profile_record` 根据变异系数、样本数和共享网络标记生成 `NONE`、`NODE_SET_QUIET` 或 `GLOBAL_QUIET` 建议。
- 性能 Step 可声明 `performance_isolation=AUTO`，由最新干扰画像决定静默范围。
- `scheduler_tuning` 给出 Agent、构建槽和候选批量建议；`scheduler_tuning_apply` 可持久化明确指定的并发参数。

## 指标与 HTML 看板

```bash
# 使用 RSI 演示数据库
PYTHONPATH=src python3 -m ascend_orchestrator --db var/rsi-demo.db metrics
PYTHONPATH=src python3 -m ascend_orchestrator --db var/rsi-demo.db dashboard --output var/dashboard.html
```

看板包含：

- 健康卡、估算 NPU 利用率、有效卡时占比和当前可避免空闲信号。
- READY 到 RUNNING 的 P50/P90 排队时间、任务完成时间和围栏次数。
- 每小时有效候选评估数、接受数、测量无效率和复测率。
- 同卡冲突、过期租约执行、错误候选接受、半组 RL 启动等正确性指标。

利用率根据原型租约和节点生命周期估算，不代替硬件遥测。当前可避免空闲是瞬时值，不是累计空闲卡时。HTML 无外部 CSS/JavaScript 依赖，可离线打开。
