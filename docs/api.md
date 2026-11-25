# HTTP API 使用说明

API 由 Python 标准库提供，默认仅监听本机：

```bash
PYTHONPATH=src python3 -m ascend_orchestrator serve --host 127.0.0.1 --port 8765
```

请求和响应均为 JSON。除 `schedule`、`recover` 外，直接写操作建议在请求体中携带全局唯一的 `request_id`。同一 ID、同一操作和同一负载会返回第一次的结果；同一 ID 对应不同操作或负载会返回 `409`。

## 通用响应

成功使用 HTTP 200。错误响应示例：

```json
{
  "error": "stale lease epoch",
  "type": "FencingError"
}
```

| HTTP 状态 | 含义 |
| --- | --- |
| 400 | 合同字段缺失、参数错误或 JSON 无效 |
| 404 | 路径或实体不存在 |
| 409 | 状态冲突、租约围栏、Owner 或产物一致性门禁拒绝 |
| 500 | 未预期的服务端错误 |

## 只读接口

| 方法与路径 | 说明 |
| --- | --- |
| `GET /health` | 服务健康和模拟 Runner 标识 |
| `GET /cluster` | 节点、健康卡、队列、预留与可避免空闲 |
| `GET /tasks/{task_id}` | Task、Steps 和 EvidencePack |
| `GET /evidence/{evidence_id}` | 指定证据包 |
| `GET /reservations/{reservation_id}` | 预留及阻塞租约 |
| `GET /events` | 追加式审计事件 |
| `GET /capabilities` | 已注册领域能力 |

## 集群与调度

| 方法与路径 | 主要字段 |
| --- | --- |
| `POST /executors/register` | `node_id`、`npus`、`labels`、`build_slots` |
| `POST /executors/heartbeat` | `node_id`、`health`、可选 `npus` |
| `POST /nodes/lifecycle` | `node_id`、`lifecycle=READY/CORDONED/DRAINING/LOST` |
| `POST /nodes/reconciled` | `node_id`、`health=HEALTHY` |
| `POST /schedule` | 无必填字段，扫描 READY DAG 并原子分配 |
| `POST /reservations` | `step_id`、`starts_at`、`guard_seconds` |
| `POST /recover` | 围栏旧执行并进入 Executor 对账状态 |

注册节点：

```bash
curl -s -X POST http://127.0.0.1:8765/executors/register \
  -H 'Content-Type: application/json' \
  -d '{
    "node_id":"node-0",
    "npus":[{"id":0},{"id":1}],
    "labels":{"soc":"910C","runtime_profile":"a"},
    "build_slots":2,
    "request_id":"register-node-0-v1"
  }'
```

## 任务与 Step

| 方法与路径 | 主要字段 |
| --- | --- |
| `POST /tasks` | `objective`、可选轨道、`scenario`、`model_spec_id`、`rl_spec_id` |
| `POST /tasks/steps` | `task_id`、`steps` |
| `POST /tasks/missing-operators` | `parent_step_id`、`operators` |
| `POST /tasks/{task_id}/pause` | 停止新步骤准入 |
| `POST /tasks/{task_id}/resume` | 恢复调度 |
| `POST /tasks/{task_id}/cancel` | 取消任务 |
| `POST /steps/complete` | 非运行中步骤的 Worker 结果 |
| `POST /steps/run` | 受控 Runner 执行及证据生成 |

创建自动识别轨道的任务：

```bash
curl -s -X POST http://127.0.0.1:8765/tasks \
  -H 'Content-Type: application/json' \
  -d '{"objective":"开发并验证 AscendC 算子","request_id":"task-001"}'
```

附加单卡 Step 时，`resource` 支持 `scope`、`nodes`、`cards_per_node`、`isolation`、`performance_isolation`、`node_selector` 和 `duration_seconds`。

执行运行中的 Step 需要从其租约读取 `lease_id` 和 `epoch`：

```json
{
  "step_id": "step-...",
  "lease_id": "lease-...",
  "epoch": 1,
  "correctness": "PASSED",
  "simulated": true,
  "request_id": "execution-001"
}
```

多机步骤还必须提交一致的 `artifact_digests`。传入 `simulated=false` 会明确失败，因为原型未配置真实 Executor。

## 模型产物

| 方法与路径 | 说明 |
| --- | --- |
| `POST /models/specs` | 注册模型、源摘要、工作负载与目标拓扑 |
| `POST /models/artifacts` | 注册派生产物及转换链摘要 |

`model` 至少包含 `name`、`source_revision`、`framework`、`mode`、`weights_digest`、`config_digest`。派生产物请求包含 `model_spec_id`、`conversion`、`content_digest` 和可选 `kind`。

## RL 契约与联合闭环

| 方法与路径 | 说明 |
| --- | --- |
| `POST /rl/specs` | 注册权重、轨迹和生成契约 |
| `POST /rl/freeze` | 冻结契约并解除双分支依赖 |
| `POST /rl/update` | 升级 `contract_version` 并同步集成步骤 |
| `POST /resource-groups` | 为 READY 集成 Step 创建原子角色资源组 |

创建 RLSpec 必须包含：

```json
{
  "weight_exchange_contract": {"format": "safetensors"},
  "trajectory_contract": {"schema_version": "v1"},
  "generation_contract": {"sampling_parameters": {"temperature": 1.0}},
  "integration_acceptance": {"rollout_train_closed_loop": "REQUIRED"}
}
```

创建 RL 任务时使用 `scenario=REINFORCEMENT_LEARNING` 和返回的 `rl_spec_id`。双方功能证据就绪后创建资源组：

```json
{
  "step_id": "step-...",
  "owner": "rl-integration-owner",
  "roles": {
    "training": {"scope": "SINGLE_NODE", "cards_per_node": 2, "lease_class": "SERVICE"},
    "inference": {"scope": "SINGLE_NODE", "cards_per_node": 2, "lease_class": "SERVICE"}
  }
}
```

联合执行除组内有效租约外，还要求一致的 `contract_versions`，以及 `metrics` 中以下证据字段：

- `weight_export=true`
- `weight_import=true`
- `trajectory_generation=true`
- `trajectory_consumption=true`
- `weight_resync=true`
- `rollout_train_rounds>=2`

分支单独通过不能替代上述闭环证据。

## 租约与领域能力

| 方法与路径 | 说明 |
| --- | --- |
| `POST /leases/heartbeat` | `lease_id`、`epoch`、可选 `progress` |
| `POST /leases/release` | 租约进入清理状态 |
| `POST /leases/cleanup` | 确认清理，之后资源才可重新分配 |
| `POST /capabilities/invoke` | `capability_id`、`inputs` |

内置示例能力为 `msagent.offline.profiler-analysis` 和 `cannbot.controlled.patch-example`。适配器失败会显式记录为 `FAILED`，不会被解释成“没有发现问题”。

## 部署安全

当前 API 没有认证和 TLS，不应直接暴露到公共网络。生产使用必须在可信网络中增加认证授权、请求大小限制、远程 Executor 身份校验、设备权限隔离和敏感日志治理。
