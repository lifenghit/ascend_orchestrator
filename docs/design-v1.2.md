# 昇腾模型与算子适配智能体资源调度及 RSI 优化系统设计

## 文档信息

- 文档版本：V1.2
- 日期：2026-11-25
- 适用环境：节点数、每节点卡数和硬件标签可动态发现的昇腾集群；4 机、每机 16 卡仅为当前部署示例，不构成架构上限或固定假设
- 任务范围：模型适配、强化学习训练与推理适配、算子开发与优化，以及它们组成的混合任务
- 宿主环境：Kimi Code
- 预留扩展：msAgent、CANNBot Skills、AReaL-RSI 或兼容 RSI 编排器
- 文档定位：架构与详细设计，可作为后续原型实现、接口评审和验收测试的依据
- 修订说明：V1.1 将模型适配提升为一等任务轨道；V1.2 移除固定四机资源假设，增加动态集群、强化学习训练／推理并行适配、交互契约、联合资源组和 RL 闭环验收

## 1. 背景与问题定义

### 1.1 业务背景

当前任务分为两条一等业务轨道，以及由它们组合形成的混合任务：

1. **模型适配轨道**：将已有模型迁移到昇腾环境，处理框架与运行时接入、权重和配置加载、图或模块转换、算子覆盖、精度对齐、单卡与分布式功能验证、显存与并行策略调整，以及模型级性能优化。模型适配既包括推理，也包括训练或微调；二者的正确性证据和性能指标允许不同。强化学习场景进一步拆成训练与推理两个可先独立适配的子轨道，通过版本化交互契约和最终 RL 闭环集成。
2. **算子工程轨道**：进行普通或通信算子的开发、精度调试、功能联调和性能优化。多数普通算子步骤只需单卡，适合多个子 Agent 并行；通信验证可能需要单机多卡或多机多卡。
3. **混合任务**：模型适配中发现缺失算子、精度瓶颈或性能热点时，在同一个 Task DAG 中创建算子子任务；算子完成后返回模型集成验证，而不是形成两套彼此割裂的系统。

性能优化相应分为三个层级：

- **单算子优化**：以冻结基线、正确性门禁和真实微基准为基础进行迭代。
- **模型级优化**：围绕固定模型与固定拓扑优化图、算子组合、显存、通信和并行策略，验证模型自身的功能与性能收益。
- **端到端优化**：以真实应用、推理服务或训练工作负载为依据，决定模型和算子候选在实际预处理、调度、并发及服务约束下是否真正产生收益。

资源总量由动态集群清单决定。在当前 4 机 64 卡示例中，如果采用静态分区，容易在某些阶段出现分区闲置；在节点扩容、缩容或健康状态变化时，静态假设还会直接失效。如果完全由 Agent 自行抢卡，又会产生冲突、环境污染和分布式任务互相破坏。系统需要解决的核心问题是：

> 在机器有限的情况下，由 AI 根据任务进展自动识别阶段和资源需求，由确定性控制面统一仲裁资源，让所有可安全并行的任务尽量填满空闲资源，减少可避免的等待和空转，同时保证通信联调与性能测量的正确性、可复现性和可回退性。

### 1.2 核心设计命题

系统不以“同时启动尽可能多的 Agent”为目标，而以“在满足正确性、隔离和公平性的前提下，尽可能减少可避免的资源空闲”为目标。

NPU 卡数只是资源维度之一。系统还需要考虑：

- 节点和卡动态加入、退出、隔离及恢复健康带来的资源视图变化；
- 不同节点或资源池的硬件、运行时和拓扑标签差异；
- 主机 CPU 和内存形成的构建瓶颈；
- 单机多卡和多机任务的拓扑需求；
- 性能测量对噪声和共享网络干扰的敏感性；
- 工作区、构建目录、日志目录和缓存的隔离；
- 模型权重、配置、分片和转换产物的一致性，以及大模型产物的存储与分发成本；
- 模型加载、图编译、权重转换和分布式初始化形成的额外准备开销；
- Agent 上下文、模型调用和结果整合能力；
- 长任务排空对分布式任务等待时间的影响；
- 任务切换、取消和重复构建带来的额外开销。

### 1.3 设计目标

1. **模型与算子统一编排**：模型适配、强化学习适配、算子开发和混合任务共用 Task DAG、调度、证据和资源控制协议。
2. **训练与推理解耦并行**：强化学习场景先冻结交互契约，训练与推理分别适配、验证和优化，只在明确的交换与闭环集成点汇合。
3. **动态集群**：节点数、卡数、硬件标签和健康状态由资源发现确定，支持节点上线、下线、扩缩容和异构资源选择。
4. **自动阶段决策**：无需用户通过斜杠命令指定单卡、单机或多机模式，由主 Agent 根据任务轨道、当前阶段和下一实际动作自动判断。
5. **资源利用率优先**：采用工作保持、动态回填、按步骤租约和批量实验窗口，减少可避免的空闲卡时。
6. **通信任务单一 Owner**：同一个单机或多机通信任务只能有一个主 Agent 决策和发起执行。
7. **全局资源一致性**：所有 NPU 执行经过统一控制面，不因模型误判或 Hook 失效而越权。
8. **RSI 原生支持**：单算子、模型级、RL 训练／推理分支与端到端优化均形成假设、候选、真实实验、比较、接受或回退的闭环。
9. **能力可插拔**：msAgent、CANNBot Skills 通过稳定适配器接入，不侵入资源调度核心。
10. **可恢复、可审计**：任务、租约、模型产物、RL 交互契约、候选、环境指纹和实验结果均持久化，支持故障恢复和复盘。

### 1.4 非目标

- 不追求理论上的全局最优调度；采用可解释的在线启发式策略。
- 不将资源利用率等同于有效研发产出，必须同时考察任务完成时间和有效实验吞吐。
- 不允许为了填卡而破坏性能测量的隔离要求。
- 不在第一阶段实现任意嵌套、多 Leader 的递归 RSI。
- 不要求 msAgent 或 CANNBot Skills 理解本系统的内部调度细节。

## 2. 总体设计原则

### 2.1 AI 决策与确定性仲裁分离

AI 负责：

- 理解用户目标；
- 拆分任务和建立依赖关系；
- 判断当前阶段和下一阶段；
- 提出资源需求及理由；
- 生成候选、分析证据和形成研究假设；
- 决定何时并行、何时收敛、何时降级。

控制面负责：

- 校验阶段转换和资源请求；
- 分配、排队、排空和回收资源；
- 保证互斥和拓扑约束；
- 执行租约心跳、超时和围栏检查；
- 进行工作区、设备和运行目录隔离；
- 记录不可篡改的执行事实。

AI 不直接修改资源状态，控制面也不替 AI 判断代码优化方向。

### 2.2 任务阶段与资源形态正交

“单算子”不等于“单卡”，“模型适配”也不等于一开始就占用完整多机集群，“多机联调”同样不意味着整个任务永久独占集群。

任务阶段描述研发进度；资源请求描述某个具体步骤需要何种资源。一个通信算子可以在单卡阶段修改本地计算逻辑，在单机阶段验证通信语义，在多机阶段验证跨机路径，并在问题缩小后退回单卡。一个模型适配任务也应先通过静态分析、配置转换和最小样例确定问题，再按模型规模、并行策略和验证目标逐步扩大到单卡、单机或多机；问题缩小后同样允许降级。

### 2.3 按步骤持有资源，而不是按会话持有资源

代码阅读、假设生成、结果分析时不持有 NPU 租约。只有实际构建、测试、profiling 或 benchmark 时才申请资源，完成后立即释放。

长会话不等于长租约。RSI Campaign 可以持续很久，但其每轮实验分别申请资源；多机性能测量可使用有时间上限的实验窗口。

### 2.4 工作保持与受控回填

默认采用工作保持策略：只要存在与空闲资源兼容、不会破坏即将开始的独占任务的排队步骤，就允许调度，不因低优先级或未来可能的大任务而主动空置资源。

当大任务已有明确预留时，仍允许回填满足以下条件之一的小任务：

- 预计在预留开始前完成；
- 可在检查点安全暂停；
- 可由控制面在截止时间前取消，且取消成本可接受。

### 2.5 分布式任务单 Owner，不等于集群单 Agent

为提高整体利用率，“多机联调只用单 Agent”定义为：

- 同一个分布式任务只有一个决策 Owner；
- 只有一个 Agent 能修改该任务的运行配置、启动 rank、停止任务和判定结果；
- 其他不冲突节点上的独立单卡任务可以继续运行；
- 性能测量需要排除干扰时，可扩大为节点集合静默或全局静默窗口。

如果将“单 Agent”解释为整个集群只能运行一个 Agent，非参与节点也会被闲置，与系统核心目标冲突。

### 2.6 真实证据优先

功能正确性和性能接受必须基于真实执行结果。代码解释、Agent 自报性能和文字结论不能替代测试报告、环境指纹、构建产物摘要和原始日志。

## 3. 总体架构

```text
用户
  │
  ▼
Kimi Code 主 Agent：Ascend Orchestrator
  ├── 目标理解与任务 DAG
  │     └── RL：RL Contract → 训练子 DAG ∥ 推理子 DAG → 闭环集成
  ├── 阶段判断与转换申请
  ├── 子 Agent 生命周期
  ├── 结果集成与用户反馈
  │
  ├───────────────► RSI Leader
  │                   ├── 单算子优化 Campaign
  │                   ├── 模型／RL 训练／RL 推理优化 Campaign
  │                   ├── 端到端优化 Campaign
  │                   ├── 候选谱系与经验
  │                   └── Comparison Contract
  │
  ├───────────────► 领域能力适配层
  │                   ├── msAgent Adapter
  │                   ├── CANNBot Skill Adapter
  │                   └── 自定义工具 Adapter
  │
  │ MCP / API
  ▼
Ascend Control Plane
  ├── Task & Phase Service
  ├── Scheduler & Backfiller
  ├── Lease & Fencing Manager
  ├── Reservation & Quiet-window Manager
  ├── Campaign Metadata Service
  ├── Artifact / Evidence Registry
  └── Event Log & Metrics
  │
  ├────────► node-0 Executor：动态卡清单、构建槽、工作区
  ├────────► node-1 Executor：动态卡清单、构建槽、工作区
  ├────────► ……
  └────────► node-N Executor：动态卡清单、构建槽、工作区

资源发现：节点加入／退出、健康、硬件标签、运行时标签、拓扑和卡数量
```

### 3.1 组件职责

| 组件 | 核心职责 | 明确不负责 |
|---|---|---|
| Ascend Orchestrator | 任务 DAG、阶段判断、并行编排、结果集成 | 直接指定物理卡、绕过控制面执行 |
| Single-card Worker | 在独立工作区完成一个单卡工作包 | 自行升级多卡、多机或继续递归派发 |
| Model Adaptation Worker | 完成模型分析、转换、适配模块或独立测试工作包 | 修改共享源权重、绕过 ModelSpec 验收或自行扩大拓扑 |
| Distributed Owner | 对一个通信或分布式模型任务统一修改、启动、分析、停止 | 同时管理另一个相互干扰的分布式任务 |
| RL Integration Owner | 在最小集成和闭环验收时统一控制训练、推理、权重交换与轨迹交换 | 在独立适配阶段阻塞双分支，或绕过契约版本 |
| RSI Leader | 假设、候选、比较、预算、经验和停止条件 | 直接管理设备、无限嵌套 Campaign |
| Control Plane | 调度、互斥、租约、预留、审计和恢复 | 判断代码优化方向 |
| Node Executor | 受控构建、运行、日志采集和清理 | 自主决定调度优先级 |
| Domain Adapter | 调用 msAgent、CANNBot Skills 并标准化结果 | 复制其领域知识或另建资源调度器 |

## 4. 核心数据模型

### 4.1 Task

Task 表示用户的一个完整目标，可包含多个可并行步骤。支持三类任务：

- `OPERATOR_DEVELOPMENT`：算子开发、修复、联调与优化；
- `MODEL_ADAPTATION`：模型导入、环境适配、功能与精度验证、分布式适配和性能优化；通过 `scenario=REINFORCEMENT_LEARNING` 启用训练／推理并行子模板；
- `MIXED`：模型、RL 分支与算子子任务共同组成的 DAG。

任务轨道只决定允许的阶段模板和证据要求，不改变统一的资源调度协议。

```yaml
task_id: task-01
task_kind: MODEL_ADAPTATION
scenario: GENERAL                 # GENERAL / REINFORCEMENT_LEARNING
objective: 完成某模型在昇腾环境中的功能适配和性能优化
repo:
  url: ...
  base_commit: ...
model_spec_id: model-spec-01
rl_spec_id: null                  # RL 任务必填
priority: normal
current_phase: MODEL_ADAPT
owner_agent: ascend-orchestrator
acceptance:
  load_and_run_required: true
  correctness_required: true
  distributed_verification_required: true
  model_performance_required: true
  e2e_performance_required: false
budget:
  max_operator_rounds: 20
  max_model_rounds: 15
  max_e2e_rounds: 10
  max_exclusive_minutes: 120
state: ACTIVE
```

### 4.2 Step

Step 是实际调度和执行的最小单位。Task 的并行性由 Step DAG 表达，而不是简单按 Task 数量判断。

```yaml
step_id: step-017
task_id: task-01
kind: OPERATOR_BENCHMARK
dependencies:
  - step-016
workspace:
  base_commit: ...
  patch_set: ...
execution:
  runner: single-card-test
  timeout_seconds: 900
resource:
  scope: SINGLE_CARD
  nodes: 1
  cards_per_node: 1
  isolation: DEVICE_EXCLUSIVE
  performance_isolation: NONE
scheduling:
  priority: normal
  preemptible: true
  checkpointable: false
  estimated_duration_seconds: 300
outputs:
  - correctness_report
  - performance_report
  - environment_fingerprint
```

### 4.3 ResourceSpec

资源范围、资源数量和隔离要求分别表达：

- `scope`：`NONE`、`CPU_ONLY`、`SINGLE_CARD`、`SINGLE_NODE`、`MULTI_NODE`。
- `isolation`：`SHARED_CPU`、`DEVICE_EXCLUSIVE`、`NODE_SET_EXCLUSIVE`。
- `performance_isolation`：`NONE`、`NODE_SET_QUIET`、`GLOBAL_QUIET`。
- `topology_constraints`：节点亲和、卡数、通信拓扑和软件环境标签。
- `node_selector`：按硬件、运行时、资源池和健康标签选择节点，不绑定固定节点名称。
- `resource_version`：调度匹配所依据的动态清单版本。
- `duration_hint`：来自历史数据或用户配置的时长估计，不作为硬事实。

### 4.4 Lease

```json
{
  "lease_id": "lease-9f31",
  "epoch": 27,
  "task_id": "task-01",
  "step_id": "step-017",
  "resources": ["node-1:npu-05"],
  "granted_at": "...",
  "expires_at": "...",
  "heartbeat_timeout_seconds": 60,
  "owner": "worker-017",
  "status": "ACTIVE"
}
```

每次执行请求必须携带 `lease_id`、`epoch` 和 `step_id`。控制面拒绝过期、已释放或围栏纪元不匹配的请求。

### 4.5 EvidencePack

所有领域工具和子 Agent 的结果统一归一化为证据包：

```yaml
evidence_id: ev-...
task_id: task-01
step_id: step-017
candidate_id: candidate-...
task_role: GENERAL            # GENERAL / TRAINING / INFERENCE / RL_INTEGRATION
contract_version: null         # RL 交互相关证据必填
source:
  kind: CANNBOT_SKILL
  name: ...
  version: ...
code:
  base_commit: ...
  patch_digest: ...
artifact_digest: ...
environment_fingerprint: ...
correctness:
  status: PASSED
  report_uri: ...
performance:
  protocol_digest: ...
  metrics_uri: ...
raw_logs_uri: ...
limitations: []
```

### 4.6 Campaign

Campaign 表示一次有预算、有基线、有停止条件的 RSI 优化过程。Campaign 与 Task 关联，但不长期占有设备。

```yaml
campaign_id: camp-01
level: OPERATOR               # OPERATOR / MODEL / E2E
role: GENERAL                 # GENERAL / TRAINING / INFERENCE / RL_INTEGRATION
leader: rsi-leader-01
baseline:
  evidence_id: ...
comparison_contract:
  primary_metric: operator_score
  direction: MAXIMIZE
  correctness_gate: REQUIRED
  constraints:
    - no_functional_regression
budget:
  max_rounds: 20
  max_no_improvement_rounds: 5
  max_wall_time_minutes: 240
status: ACTIVE
```

### 4.7 ModelSpec 与模型产物

模型适配任务必须额外记录模型及其工作负载，避免将“某次能运行”误当成“目标模型已经适配”。

```yaml
model_spec_id: model-spec-01
model:
  name: ...
  source_revision: ...
  framework: ...
  mode: INFERENCE              # INFERENCE / TRAINING / FINETUNING
  weights_digest: ...
  config_digest: ...
  tokenizer_or_preprocessor_digest: ...
target:
  runtime_profile: ...
  minimum_hardware_scope: SINGLE_CARD
  intended_hardware_scope: MULTI_NODE
workload:
  dataset_or_request_digest: ...
  input_shapes: ...
  precision_policy: ...
  parallelism_policy: ...
adaptation:
  entrypoint: ...
  conversion_pipeline: ...
  custom_operators: []
acceptance:
  functional_cases: []
  accuracy_contract: ...
  stability_contract: ...
```

模型产物分为：

- **只读源产物**：原始权重、配置、分片、预处理器和参考结果，以内容摘要标识；
- **派生产物**：转换后权重、图、引擎、并行切分、编译缓存和适配配置；
- **运行证据**：加载结果、算子覆盖、输出或训练指标、显存、通信、性能和稳定性报告。

多个步骤可共享只读源产物，但不得并发改写同一派生产物。派生产物必须记录源产物摘要、转换代码版本和参数；源模型或转换链路变化后，旧派生产物不能无说明地继续作为同一基线使用。大体积权重优先采用共享只读存储、内容寻址引用或受控分发，避免每个 Worker 复制一份。

### 4.8 RLSpec 与训练／推理交互契约

强化学习适配任务通过 `RLSpec` 描述两个独立分支及其最小耦合面：

```yaml
rl_spec_id: rl-spec-01
contract_version: v1
base_model:
  model_spec_id: model-spec-01
  weights_digest: ...
  tokenizer_or_preprocessor_digest: ...
training_branch:
  task_id: task-train-01
  model_spec_id: model-spec-train-01
  entrypoint: ...
  reference_checkpoint: ...
  acceptance:
    forward_backward: REQUIRED
    optimizer_update: REQUIRED
    checkpoint_save_restore: REQUIRED
    convergence_contract: ...
inference_branch:
  task_id: task-infer-01
  model_spec_id: model-spec-infer-01
  entrypoint: ...
  reference_weights: ...
  acceptance:
    load_and_generate: REQUIRED
    sampling_contract: ...
    weight_import: REQUIRED
weight_exchange_contract:
  export_format: ...
  import_format: ...
  tensor_layout: ...
  dtype_mapping: ...
  sharding_and_resharding: ...
  version_handshake: ...
trajectory_contract:
  schema_version: ...
  observation_action_schema: ...
  log_probability_fields: ...
  mask_reward_and_metadata: ...
generation_contract:
  sampling_parameters: ...
  sequence_limits: ...
integration_acceptance:
  weight_roundtrip: REQUIRED
  trajectory_roundtrip: REQUIRED
  rollout_train_closed_loop: REQUIRED
```

训练和推理分支分别使用冻结的参考权重、参考 checkpoint、轨迹样例和生成配置进行独立适配。契约变化必须提升 `contract_version` 并显式更新受影响步骤；任何一侧不得静默修改张量布局、权重版本、采样语义或轨迹字段。

分支独立通过只表示具备集成条件，不表示 RL 系统已验收。最终还必须验证训练导出到推理导入、推理轨迹到训练消费，以及目标 rollout—train 闭环。

### 4.9 ResourceGroupSpec 与长运行角色

训练和推理独立适配时分别申请普通 Step 资源，不建立联合资源组。只有最小闭环、联合 profiling 或端到端测量确实要求两类角色同时在线时，才使用 `ResourceGroupSpec`：

```yaml
resource_group_id: rg-rl-01
purpose: RL_ROLLOUT_TRAIN
allocation_mode: ATOMIC          # 全部满足或全部不分配
roles:
  training:
    scope: MULTI_NODE
    nodes: 2
    cards_per_node: 16
    node_selector:
      runtime_profile: profile-a
  inference:
    scope: SINGLE_NODE
    nodes: 1
    cards_per_node: 8
    node_selector:
      runtime_profile: profile-a
lifetime:
  lease_class: SERVICE
  max_duration_seconds: 7200
  renewable: true
coordination:
  start_barrier: REQUIRED
  stop_barrier: REQUIRED
  weight_exchange_channel: ...
```

联合资源组必须原子分配或在超时后整体回滚，防止训练持有部分资源等待推理、推理又持有另一部分资源等待训练而形成死锁。长期在线角色使用有最大时长、可续租的 `SERVICE` 租约，并持续上报心跳；它们不能借长期运行绕过公平性和预算。

资源角色可以位于不相交节点，也可以在用户明确允许且测量隔离满足时共享节点。任何共享都不能绕开单卡互斥与静默约束。

## 5. 任务生命周期与 AI 阶段决策

### 5.1 阶段定义

| 阶段 | 适用轨道 | 主要工作 | 默认并行策略与资源特征 |
|---|---|---|---|
| `ANALYZE` | 全部 | 阅读需求、代码、模型配置和测试，形成 DAG | 只读探索可并行，通常不占 NPU |
| `RL_CONTRACT_FREEZE` | RL、混合 | 冻结权重交换、轨迹、采样、版本握手和分支验收契约 | 不占卡，完成后才能解除双分支并行依赖 |
| `ENVIRONMENT_PREPARE` | 模型、RL、混合 | 准备运行时、依赖、入口和最小环境 | 独立准备可并行，优先不占卡 |
| `MODEL_ADAPT` | 模型、混合 | 处理权重、配置、图或模块转换、接口和适配代码 | 独立模块可并行，转换按 CPU 与存储配额执行 |
| `TRAINING_ADAPT` | RL、混合 | 训练加载、前反向、优化器、checkpoint 和训练分布式适配 | 与推理分支并行，使用自己的最小拓扑 |
| `INFERENCE_ADAPT` | RL、混合 | 推理加载、生成、采样、权重导入和推理分布式适配 | 与训练分支并行，使用自己的最小拓扑 |
| `COVERAGE_VERIFY` | 模型、混合 | 检查模型加载、算子覆盖、缺失实现和不支持路径 | 静态检查不占卡；动态覆盖按最小样例申请 |
| `IMPLEMENT` | 算子、混合 | 实现或修复算子，补充测试 | 独立工作包可并行，编码不占卡 |
| `FUNCTIONAL_VERIFY` | 全部 | 验证功能、输出、精度、训练行为和通信语义 | 按验证目标选择单卡、单机或多机 |
| `DISTRIBUTED_VERIFY` | 模型、RL、混合 | 验证并行切分、权重分片、通信、初始化和跨 rank 一致性 | 同一分布式验证单 Owner |
| `RL_INTEGRATION_VERIFY` | RL、混合 | 验证权重交接、轨迹交接、版本握手和 rollout—train 最小闭环 | 训练与推理 Owner 收敛为一个集成 Owner |
| `OPERATOR_BASELINE` | 算子、混合 | 冻结算子基线、输入、协议和环境 | 短租约 |
| `OPERATOR_OPTIMIZE` | 算子、混合 | 单算子 RSI 优化 | 单卡候选可并行，通信测量单 Owner |
| `MODEL_BASELINE` | 模型、RL、混合 | 冻结模型产物、固定拓扑、模型工作负载和性能协议；RL 按 TRAINING／INFERENCE 分支分别冻结 | 每个分支独立测量 Owner |
| `MODEL_OPTIMIZE` | 模型、RL、混合 | 模型级 RSI；RL 可分别优化训练与推理分支 | 候选生成可并行，测量受各分支拓扑与隔离约束 |
| `INTEGRATION_VERIFY` | 全部 | 将算子或模型候选放回目标框架和实际模型验证 | 由依赖关系决定 |
| `E2E_BASELINE` | 全部 | 冻结真实应用负载和端到端基线 | 按真实部署形态申请 |
| `E2E_OPTIMIZE` | 全部 | 端到端 RSI 优化 | 批量实验窗口 |
| `FINAL_REGRESSION` | 全部 | 最终功能、精度、性能、稳定性和回退检查 | 独立测试可并行 |
| `DONE` | 全部 | 归档、清理和交付 | 释放全部资源 |

不同任务轨道使用不同的阶段模板，控制面按模板校验转换，不要求所有任务经过全部阶段。

**算子轨道：**

```text
ANALYZE → IMPLEMENT → FUNCTIONAL_VERIFY
  → OPERATOR_BASELINE → OPERATOR_OPTIMIZE
  → INTEGRATION_VERIFY → 可选的 E2E 阶段 → FINAL_REGRESSION
```

**模型适配轨道：**

```text
ANALYZE → ENVIRONMENT_PREPARE → MODEL_ADAPT → COVERAGE_VERIFY
  → FUNCTIONAL_VERIFY → 按需要进入 DISTRIBUTED_VERIFY
  → MODEL_BASELINE → MODEL_OPTIMIZE
  → INTEGRATION_VERIFY → 可选的 E2E 阶段 → FINAL_REGRESSION
```

**强化学习适配轨道：**

```text
ANALYZE → RL_CONTRACT_FREEZE
  ├── TRAINING_ADAPT
  │     → 训练 COVERAGE_VERIFY / FUNCTIONAL_VERIFY
  │     → 按需要进入训练 DISTRIBUTED_VERIFY
  │     → 训练 MODEL_BASELINE / MODEL_OPTIMIZE
  │
  ├── INFERENCE_ADAPT
  │     → 推理 COVERAGE_VERIFY / FUNCTIONAL_VERIFY
  │     → 按需要进入推理 DISTRIBUTED_VERIFY
  │     → 推理 MODEL_BASELINE / MODEL_OPTIMIZE
  │
  └── 两个分支仅在最小时序和契约上耦合
        → RL_INTEGRATION_VERIFY
        → E2E_BASELINE / E2E_OPTIMIZE
        → FINAL_REGRESSION
```

两个分支的适配和基础验证应并行推进。`RL_INTEGRATION_VERIFY` 只依赖双方达到最小可集成条件，不依赖另一方已经完成全部性能优化。任一分支发现内部问题时，另一分支继续推进；涉及交互契约的修改则必须通过新版本契约显式同步。

**混合反馈路径：**

```text
COVERAGE_VERIFY / FUNCTIONAL_VERIFY / MODEL_OPTIMIZE
  → 发现缺失、错误或瓶颈算子
  → 创建 OPERATOR_DEVELOPMENT 子 DAG
  → 算子证据和产物返回模型任务
  → 回到 MODEL_ADAPT、FUNCTIONAL_VERIFY 或 MODEL_OPTIMIZE
```

允许跳过不必要阶段。例如模型已经能在目标单卡环境运行时，不应机械地重新执行全部转换步骤；模型只做单卡适配时，不应强制进入多机验证。

### 5.2 阶段转换入口

主 Agent 在以下事件发生时重新评估阶段，而不是每条命令都调用模型：

1. 新任务创建，主 Agent 判定任务轨道和验收目标；
2. 分析、环境准备、模型转换或实施计划完成；
3. 一个工作包完成或失败；
4. 模型加载、算子覆盖、功能、精度或分布式测试暴露新的问题层级；
5. 单算子、模型级或端到端 Campaign 达到接受或停止条件；
6. 算子子 DAG 完成并返回模型任务；
7. 集成验证完成；
8. 用户改变目标、优先级或时间约束；
9. 会话从持久化状态恢复。

```json
{
  "task_id": "task-01",
  "from_phase": "FUNCTIONAL_VERIFY",
  "to_phase": "OPERATOR_OPTIMIZE",
  "reason": "功能与精度门禁通过，已形成冻结基线",
  "evidence_ids": ["ev-functional-01", "ev-baseline-01"],
  "expected_next_steps": ["candidate-generation", "candidate-evaluation"]
}
```

控制面校验必要证据、当前活动和资源状态后接受或拒绝转换。阶段转换不要求用户手动选择资源模式。

### 5.3 AI 决策约束

1. 先判定任务轨道和验收目标，再根据下一步实际动作申请资源，不只根据模型或算子名称判断。
2. 使用能够覆盖当前验证目标的最小资源集合；模型适配从最小可行运行开始，不因目标最终为多机而提前占用多机。
3. 信息不足时先执行不占卡的代码、权重配置、转换链路、测试入口和并行策略分析。
4. 子 Agent 只能建议阶段升级，不能自行扩大资源范围。
5. 模型适配发现算子问题时创建独立子 DAG；只有存在依赖的功能验证等待该子 DAG，其他可独立进行的模型检查继续执行。
6. RL 任务先冻结交互契约，再并行训练与推理子 DAG；不得把一侧的全部适配和优化设置为另一侧开始的依赖。
7. RL 任一分支内部阻塞不暂停另一分支；只有契约变更、最小集成和最终闭环形成显式同步点。
8. 多机问题缩小到单卡后，应主动降级并释放独占资源。
9. 不以“提高利用率”为理由运行与目标无关的任务。
10. 性能优化必须从正确性门禁通过后的冻结基线开始；模型级基线还需绑定权重、配置、派生产物和并行拓扑。

### 5.4 子 Agent 升级协议

当单卡 Worker 发现需要多卡或多机时，返回：

```json
{
  "status": "BLOCKED_BY_RESOURCE_SCOPE",
  "recommended_phase": "FUNCTIONAL_VERIFY",
  "recommended_scope": "SINGLE_NODE",
  "reason": "下一测试需要单机多 rank 通信",
  "workspace": "...",
  "commit": "...",
  "completed_evidence_ids": ["..."]
}
```

主 Agent 保存工作现场并统一申请阶段和资源转换。Worker 不直接启动分布式进程。

## 6. 资源抽象与隔离模型

### 6.1 资源清单

每个节点维护以下动态信息：

```yaml
node_id: node-0
resource_version: rv-182
lifecycle: READY                # JOINING / READY / CORDONED / DRAINING / LOST
discovered_at: ...
last_heartbeat_at: ...
labels:
  soc: 910C
  cann_profile: profile-a
  runtime_profile: profile-a
  resource_pool: default
npus:
  - id: 0
    health: HEALTHY
    lease_id: null
    exclusive_group: null
cpu:
  total_slots: ...
  build_slots: 2
  active_builds: 0
memory:
  scheduler_limit: ...
storage:
  workspace_root: ...
  artifact_root: ...
network:
  interface_profile: ...
topology:
  inventory_digest: ...
```

节点信息来自执行器周期上报和调度前预检。健康检查失败、版本不匹配或上报过期的节点不可分配。控制器不内置节点总数、每节点卡数或总卡数；资源请求使用数量、标签和拓扑约束，具体物理节点由调度器选择。

### 6.2 动态集群与资源版本

- **节点加入**：Executor 注册后进入 `JOINING`，完成设备、环境、拓扑和工作区检查才转为 `READY`，随后自动进入调度池。
- **节点扩缩容**：每次资源清单变化递增 `resource_version`，调度结果记录所依据的版本；分配时版本不一致则重新匹配，避免按过期视图分配。
- **节点隔离**：`CORDONED` 只禁止新任务，已有任务继续；`DRAINING` 停止准入并等待安全退出或受控抢占。
- **节点退出与失联**：停止新分配并围栏旧租约，未确认清理前不得把同名节点直接视为健康资源重新加入。
- **异构选择**：通过硬件、CANN 环境、运行时、资源池和拓扑标签匹配，不假设所有节点可以互换。
- **静态清单兼容**：小型部署允许提供静态种子配置，但运行中的实际数量仍以 Executor 发现与健康状态为准。

### 6.3 四类互斥关系

1. **设备互斥**：同一物理卡同一时间只能有一个设备独占租约。
2. **节点集合互斥**：单机或多机通信任务独占其资源集合，冲突的单卡任务不能新进入。
3. **测量静默互斥**：性能任务要求 `NODE_SET_QUIET` 或 `GLOBAL_QUIET` 时，禁止指定范围内其他可能引入噪声的执行。
4. **工作区互斥**：同一工作区同一时刻只能有一个写入 Owner；独立候选必须使用独立 worktree 或等效隔离。

### 6.4 Agent 并发与 NPU 并发分离

```text
max_agent_workers：由模型调用、CPU、存储和主 Agent 整合能力决定
max_npu_leases：由健康空闲卡和资源互斥决定
max_build_slots：由每机 CPU、内存和编译负载决定
```

卡数增加不代表应同比增加子 Agent 数量。没有 NPU 的编码步骤可以占用 Agent 并发，实际 NPU 测试步骤再进入设备队列。当前 64 卡示例推荐初始 `max_agent_workers=8`，每机 `build_slots=2`；集群变化后根据实测吞吐调整，而非从卡数直接推导 Agent 数。

## 7. 调度器设计

### 7.1 调度目标

调度器按以下顺序优化：

1. 保证设备互斥、通信语义和测量隔离；
2. 保证高优先级和关键路径任务不被低优先级探索任务长期阻塞；
3. 减少可避免的空闲卡时；
4. 减少排队等待、资源碎片和模式切换开销；
5. 提高单位时间内完成的任务数和有效 RSI 实验数；
6. 通过老化机制避免低优先级任务永久饥饿。

调度不是单纯贪心填卡。若短任务不断进入即将用于多机联调的节点，会显著增加大任务等待；若为了未来大任务过早清空集群，又会造成资源浪费。因此采用“预留＋回填＋老化”的组合。

### 7.2 队列与状态

每个 Step 具有以下调度状态：

```text
DRAFT
  → BLOCKED          依赖未满足
  → READY            可参与调度
  → RESERVED         已绑定未来预留，尚未执行
  → ACQUIRING        正在申请租约
  → RUNNING
  → CHECKPOINTING    可抢占任务正在保存现场
  → SUCCEEDED / FAILED / CANCELLED
  → RETRYABLE        等待重试
```

调度器维护：

- `ready_queue`：所有依赖满足的步骤；
- `wait_queue`：暂时无兼容资源的步骤；
- `reservation_table`：未来独占与静默窗口；
- `running_table`：运行步骤、租约和预计结束时间；
- `campaign_queue`：等待批量实验窗口的 RSI 候选。

### 7.3 调度评分

对兼容同一资源的 READY 步骤计算可解释评分：

```text
Score = W_user    × 用户优先级
      + W_critical × 关键路径收益
      + W_unblock  × 解除依赖收益
      + W_age      × 等待老化
      + W_window   × 当前实验窗口匹配度
      + W_pack     × 资源装箱收益
      - W_preempt  × 抢占与重跑成本
      - W_switch   × 环境与模式切换成本
```

权重均可配置。该评分不直接替代硬约束：任何高分任务都不能突破设备互斥、阶段门禁或静默窗口。

### 7.4 单次调度循环

```text
1. 接收节点加入／退出、健康变化、任务、租约和阶段事件，更新资源视图及 `resource_version`。
2. 从 Task DAG 中加入新的 READY 步骤，完成依赖解除；RL 训练与推理分支分别解除依赖。
3. 处理已到期的独占预留和静默窗口，优先保证其资源确定性。
4. 过滤不满足健康、版本、拓扑、隔离和阶段门禁的步骤。
5. 按评分选择当前可执行步骤并原子创建租约；分配前重新检查资源版本，联合资源组执行全有或全无分配。
6. 对剩余空隙执行回填，寻找短任务、可检查点任务、CPU-only 任务或另一 RL 分支的独立步骤。
7. 对即将到期的长任务检查心跳、进度和抢占要求。
8. 对等待过久任务提高老化权重，必要时触发预留重估。
9. 记录调度原因、拒绝原因和资源碎片情况。
```

调度由事件触发并辅以短周期检查，不依赖 LLM 持续轮询。

### 7.5 工作保持策略

当同时满足以下条件时，空闲资源不应保持闲置：

- 队列中存在兼容的 READY 步骤；
- 该步骤不违反任何互斥或静默条件；
- 不影响已经生效的预留；
- 执行器健康且构建、存储等配套资源可用；
- 没有明确的用户暂停要求。

系统记录 `avoidable_idle_card_time`，用于发现“有任务可跑但资源空置”的调度缺陷。

### 7.6 回填策略

设某独占预留的开始时间为 `T_start`，当前时间为 `T_now`，安全清理余量为 `T_guard`，则回填步骤需要满足：

```text
预计完成时间 + 清理余量 <= 预留开始时间
即：T_now + duration_p90 + T_guard <= T_start
```

若无法可靠估计时长，可采用以下策略之一：

- 只允许已知短测试回填；
- 允许可检查点任务回填，并提前触发保存；
- 经策略允许后执行可取消任务，但记录潜在重跑成本；
- 不做有风险回填，并将空闲归类为“预留保护空闲”，而非可避免空闲。

时长估计来自同类型步骤历史分布，优先使用保守分位数，而不是依赖 Agent 主观承诺。

### 7.7 大任务预留

多机任务只有在满足以下条件后才创建正式预留：

- 功能阶段已达到需要多机的步骤；
- 代码、构建产物和运行配置已基本就绪；
- 节点数、每机卡数和隔离要求明确；
- 任务已具备执行条件，不是遥远未来的可能性。

禁止因为“后面可能需要四机”而提前空置整个集群。

预留创建后：

1. 目标资源集合停止接受不满足回填条件的新任务；
2. 现有任务继续运行到安全点、检查点或截止时间；
3. 非目标节点继续正常调度；
4. 对目标节点空隙执行受控回填；
5. 所有冲突租约释放后，预留转为 ACTIVE；
6. 超过排空期限时，上报阻塞任务并由策略选择继续等待或受控抢占。

### 7.8 资源碎片与拓扑感知

多卡任务不应只按“全集群剩余卡数”判断可执行性。例如剩余 16 卡分布在四台机器上，并不等价于一机 16 卡。

调度器需要区分：

- 单卡：可使用任意健康兼容卡；
- 单机多卡：必须在同一节点满足卡数与环境要求；
- 多机多卡：必须同时满足节点集合、每机卡数、运行时环境和通信配置；
- 性能任务：还要满足测量隔离和稳定性要求。

在满足硬约束后，可使用拓扑亲和评分选择资源集合，但不把物理卡号连续作为通用硬要求。

### 7.9 防饿死与公平性

- 等待时间越长，老化权重越高；
- 用户高优先级和多机关键路径任务可创建预留；
- RSI 探索候选默认低于功能阻塞修复，但可使用批量窗口获得稳定进展；
- 同一 Campaign 不得无限占用静默窗口；
- 每次抢占必须记录原因和成本；
- 连续被抢占的任务获得补偿优先级；
- 达到最大等待时间后，触发用户可见的调度说明，而不是无声排队。

### 7.10 默认优先级建议

```text
用户明确指定的紧急任务
  > 阻塞功能联调和关键路径验证
  > 最终回归与基线复验
  > 普通单卡开发和功能测试
  > 单算子 RSI 探索
  > 低优先级批量实验
```

该顺序只是初始配置。端到端 Campaign 已承诺实验窗口时，窗口内任务按预留执行，避免每次都被新的小任务挤占。

## 8. 模式切换与典型调度流程

### 8.1 单卡并行阶段

1. 主 Agent 将目标拆成独立工作包和依赖。
2. 编码和阅读步骤先执行，不占卡。
3. 需要测试的步骤进入 READY 队列。
4. 调度器按空闲卡、构建槽和优先级分配资源。
5. 每个 Worker 只获得自己的租约和工作区。
6. 测试完成后立即释放卡，Worker 可继续分析结果。
7. 主 Agent 收集 EvidencePack，决定继续实现、验证或进入优化。

### 8.2 单机或多机联调升级

1. 主 Agent 提交阶段转换和最小资源需求。
2. 控制面冻结该分布式任务的 Owner，禁止第二个执行 Owner。
3. 只对目标资源集合停止冲突任务准入，其他资源继续工作。
4. 已有冲突任务运行到安全点或被受控抢占。
5. 目标集合空隙可执行满足截止时间的回填任务。
6. 资源齐备后进行环境预检和统一构建。
7. Distributed Owner 发起一次受控分布式运行。
8. 结束后统一清理、归档并释放资源。
9. 调度器立即重新扫描 READY 队列并填满释放资源。

### 8.3 多机问题降级

如果多机问题已经缩小到单卡可复现问题：

1. Distributed Owner 保存多机现场和证据；
2. 停止所有 rank 并清理；
3. 释放节点集合或全局静默租约；
4. 主 Agent 将阶段退回 `IMPLEMENT` 或 `FUNCTIONAL_VERIFY`；
5. 创建单卡修复和验证工作包；
6. 其他排队任务同时回填已释放资源。

### 8.4 示例：两机联调时的资源利用

假设任务 A 需要 node-0、node-1 各 16 卡进行联调：

- node-0、node-1 进入目标集合排空；
- node-2、node-3 的 32 卡继续运行独立单卡任务；
- node-0、node-1 上已运行任务继续到安全点；
- 若预留尚未开始，可在其上回填确定能按时结束的短任务；
- A 联调期间，只有 A 的 Distributed Owner 能操作 A 的代码、rank 和运行配置；
- 若 A 只是功能联调，node-2、node-3 不需要静默；
- 若 A 的跨机性能会受其他任务共享网络影响，再按证据扩大静默范围。

### 8.5 示例：四机端到端性能窗口

假设任务 B 需要四机 64 卡，并要求 `GLOBAL_QUIET`：

- B 在候选生成和分析阶段不占卡；
- 多个候选尽量在窗口前完成构建和正确性初筛；
- 控制面为一批候选创建一个限时全局窗口，而不是每个候选各排空一次；
- 窗口前停止新冲突任务并执行受控回填；
- 窗口内只运行 B 指定的测量和必要控制任务；
- 窗口结束后保存 Campaign checkpoint，释放资源并恢复普通调度。

### 8.6 模型适配任务的并行与收敛

模型适配不能因为最终需要多机，就从任务开始独占多机。推荐按以下方式拆分：

**可并行步骤：**

- 模型代码、配置、权重结构和运行时接口的只读分析；
- 相互独立的适配模块、转换脚本和测试用例开发；
- 不互相修改共享产物的精度用例、最小功能样例和独立候选；
- 模型适配主任务等待某项结果时，可并行推进缺失算子子 DAG；
- NPU 功能测试运行期间，可并行执行其他候选的 CPU 转换、构建或日志分析。

**必须收敛的步骤：**

- 修改同一模型主配置、同一转换产物或同一工作区；
- 对同一分布式模型任务进行并行切分、rank 启动、通信验证和现场清理；
- 需要统一权重、图、引擎和运行时版本的模型级基线测量；
- 对共享存储或网络干扰敏感的性能实验。

模型适配采用“分层证据”防止错误提前验收：模块可运行不等于整模型可运行，单卡可运行不等于分布式正确，单次输出可产生不等于精度、稳定性和性能均通过。只有当前验收契约要求的层级全部具备证据，才能进入下一阶段。

对于大模型，加载、转换和分发时间纳入 Step 时长估计与回填判断。源权重尽量共享只读；每个候选只保存配置和补丁差异，派生产物按内容摘要复用。候选使用独立派生产物时，由执行器控制写入权限，避免一个候选污染另一个候选。

### 8.7 强化学习训练／推理双分支调度

**契约完成后立即解除双分支依赖。** 训练与推理分别形成 READY 步骤，由调度器按当前健康资源和优先级独立分配：

- 训练侧可执行前反向、优化器、loss、checkpoint、训练并行和训练性能步骤；
- 推理侧可执行权重加载、生成、采样、KV 或运行时、推理并行和推理性能步骤；
- 任一侧等待特定算子、数据或修复时，另一侧继续运行；
- 可共享只读模型和契约，不共享可写运行现场；
- 两侧各自发现算子问题时，可创建独立或共享依赖的算子子 DAG。

**不为未来闭环提前空留资源。** 如果训练已就绪而推理尚未达到最小集成条件，训练任务可以正常使用资源；推理侧同理。只有双方都满足 `RL_INTEGRATION_VERIFY` 前置条件后，才申请联合资源组。

**闭环集成收敛 Owner。** 训练与推理分支 Owner 将运行现场、版本和证据交给一个 RL Integration Owner。联合资源组原子就绪后，由该 Owner 统一执行：

1. 基模型与契约版本检查；
2. 训练 checkpoint 导出、推理权重导入和版本握手；
3. 推理生成与轨迹 schema 校验；
4. 训练消费轨迹并完成更新；
5. 新权重再次同步到推理侧；
6. 最小多轮 rollout—train 闭环和稳定性验证；
7. 统一停止、清理和证据归档。

最小闭环通过后，可释放联合资源组，训练与推理重新进入独立优化；最终端到端性能验收再申请限时联合窗口。这样既支持双分支并行，也避免长期占用两套在线资源。

## 9. 租约、执行与故障恢复

### 9.1 租约生命周期

```text
REQUESTED → GRANTED → ACTIVE → RELEASING → RELEASED
                    │        │
                    │        ├→ EXPIRED → FENCED → CLEANED
                    │        └→ REVOKED → CLEANED
                    └→ HEARTBEAT
```

- 所有分配在单事务中完成，避免并发重复分配。
- 执行器启动任务前再次校验租约和纪元。
- 每次构建、测试、日志写入和清理都关联 `lease_id`。
- 心跳超时后先隔离旧执行，再允许资源重新分配。
- 旧 Worker 即使恢复，也因纪元过期不能继续执行。

### 9.2 工作区隔离

每个工作包使用：

```text
workspaces/<task-id>/<step-id>/    独立代码工作区
build/<task-id>/<step-id>/         独立构建目录
runs/<task-id>/<step-id>/          独立运行与 profiling 目录
artifacts/<task-id>/<candidate>/   构建产物与证据
```

要求：

- 不并发修改同一 worktree；
- 不共享可写的 profiling 输出目录；
- 不在 Worker 内执行全局依赖安装或系统环境修改；
- 构建缓存如共享，必须内容寻址或具备并发安全；
- 多机任务使用同一代码基线、补丁摘要和构建产物摘要；
- 模型任务额外校验源权重、配置、分片、转换产物、图或引擎摘要，跨机运行前完成一致性检查；
- 共享模型源产物只读，候选的转换输出和配置修改写入独立目录；
- 设备可见性由执行器按进程注入，不依赖 Agent 自行设置。

### 9.3 执行权限

Agent 默认不通过通用 shell 直接访问 NPU。所有 NPU 操作经由受控 Runner：

- `build-runner`；
- `single-card-test-runner`；
- `single-node-test-runner`；
- `multi-node-test-runner`；
- `profile-runner`；
- `cleanup-runner`。

Runner 校验资源语义。例如单卡租约不能调用多机 Runner，多机 Runner 不能缺少对应节点集合租约。

### 9.4 故障处理矩阵

| 故障 | 控制面行为 | 恢复方式 |
|---|---|---|
| Worker 崩溃 | 租约心跳超时，冻结旧纪元 | 保存工作区，重新调度未完成步骤 |
| Executor 失联 | 节点标记不可调度，禁止新分配 | 节点恢复后先做现场与设备检查 |
| Controller 重启 | 从持久化状态恢复，不直接相信旧运行状态 | 与 Executor 对账后恢复或清理 |
| 分布式任务部分 rank 失败 | 整个运行标记失败，保留各 rank 证据 | Owner 决定重试、降级或返修 |
| 性能结果异常波动 | 不接受候选，标记测量无效 | 检查干扰并在相同协议下复测 |
| 领域适配器超时 | 只失败当前能力调用 | 可重试或选择其他能力，不释放无关任务 |
| 清理失败 | 资源保持隔离，不可立即复用 | 告警并进入人工或自动深度清理 |

控制面对资源操作采用失败关闭；无法确认资源安全时，不允许继续分配。

### 9.5 分布式运行协议

`run_multi_node_test` 不是向各节点分别发送互不关联的启动命令，而是创建一个全局 `execution_id` 和唯一运行清单：

1. Distributed Owner 提交代码、产物摘要、节点集合、rank 配置、环境摘要、超时和清理策略。
2. 控制面再次校验全部租约、Owner、隔离等级和产物一致性。
3. 各 Executor 完成运行前预检并进入 `READY_TO_LAUNCH`；任一节点失败则整个执行不启动。
4. 控制面在同一执行纪元下触发各节点启动，避免部分节点使用旧配置。
5. Executor 持续上报进程、rank、日志和心跳；部分 rank 失败时，整个执行按策略停止或进入统一诊断。
6. Owner 通过同一个 `execution_id` 查询、取消和收集证据，不分别操纵单个 rank。
7. 所有节点完成清理并确认后，相关资源才重新进入可分配池。

确定性日志解析、指标聚合和错误聚类可以由脚本并行执行，但这些脚本没有修改配置、重启 rank 或接受候选的权限。

## 10. RSI 单算子、模型级与端到端优化设计

### 10.1 RSI 在生命周期中的位置

RSI 是优化阶段的执行引擎，不是独立资源调度器。算子轨道采用：

```text
IMPLEMENT
  → FUNCTIONAL_VERIFY
  → OPERATOR_BASELINE
  → OPERATOR_OPTIMIZE（单算子 RSI）
  → INTEGRATION_VERIFY
  → 可选的 E2E_BASELINE / E2E_OPTIMIZE
  → FINAL_REGRESSION
```

模型适配轨道采用：

```text
MODEL_ADAPT / COVERAGE_VERIFY
  → FUNCTIONAL_VERIFY
  → 按需要进入 DISTRIBUTED_VERIFY
  → MODEL_BASELINE
  → MODEL_OPTIMIZE（模型级 RSI）
  → INTEGRATION_VERIFY
  → 可选的 E2E_BASELINE / E2E_OPTIMIZE
  → FINAL_REGRESSION
```

模型级 Campaign 可以把热点或缺失算子交给有预算的算子子 Campaign；端到端 Campaign 也可以将问题返回模型级或算子级。子 Campaign 只返回候选和证据，不自行宣布上层目标成功。

只要求独立算子优化时，可以在 `OPERATOR_OPTIMIZE` 后进入最终回归。只要求模型在指定拓扑下可运行时，可以在功能或分布式验证后结束，不强制进行性能 Campaign。要求模型级性能时，算子结果只是模型候选的一部分；要求真实应用收益时，最终接受权属于端到端 Campaign。

### 10.2 单一 RSI Leader

每个 Campaign 只有一个 Leader。Leader 负责：

- 维护冻结基线和 Comparison Contract；
- 提出可检验假设；
- 生成一个有向无环的候选批次；
- 控制候选预算、并行度和实验窗口需求；
- 读取真实 EvidencePack；
- 接受、拒绝、回退或要求复测；
- 维护候选谱系、失败经验和停止原因。

普通 Worker 或领域 Skill 不启动自己的 RSI 循环，避免多套基线和经验相互覆盖。

### 10.3 单算子优化闭环

```text
冻结基线
  → 提出假设
  → 生成一个或多个候选
  → 独立构建
  → 正确性门禁
  → 性能初筛
  → 受控环境复测
  → 比较与归因
  → 接受为候选或回退
  → 更新经验和下一轮假设
```

普通单卡算子可并行评估多个候选。最终头对头比较应尽量使用一致的环境和测量协议，必要时在同一验证资源上复测。

通信算子的优化资源由实验内容决定：本地计算部分可以单卡筛选；涉及多 rank 通信语义和性能时，必须使用能够覆盖该机制的最小拓扑，并由唯一 Distributed Owner 或 RSI Leader 发起。

### 10.4 单算子 Comparison Contract

必须包括：

- 正确性硬门禁；
- 固定输入 shape、dtype、数据分布和边界用例；
- 固定预热、重复、统计和异常波动处理；
- 一个主指标及方向；
- 必要的显存、功能泛化和稳定性约束；
- 基线与候选的环境指纹；
- 最小有效收益和复测规则；
- 失败候选保留要求。

单算子通过后标记 `KERNEL_CANDIDATE_ACCEPTED`，不直接声称产生端到端收益。

### 10.5 模型级优化闭环

模型级 RSI 的目标不是“让模型碰巧跑一次”，而是在固定模型、固定产物和固定拓扑下，形成可复现的功能与性能改进。

进入 `MODEL_OPTIMIZE` 前必须冻结：

- 源模型、权重、配置、预处理器和参考结果摘要；
- 转换链路、派生权重、图或引擎产物摘要；
- 模型代码和适配补丁；
- 输入、数据或请求集合；
- 并行策略、rank 拓扑和运行时版本；
- 功能、精度、训练行为、显存和稳定性门禁；
- 一个模型级主指标及测量协议。

模型级动作空间包括：

1. `OPERATOR_SET`：选择、替换或组合算子候选；
2. `GRAPH_ONLY`：调整图转换、融合、常量折叠或执行路径；
3. `MEMORY_ONLY`：调整显存、重计算、缓存、权重驻留或加载策略；
4. `PARALLEL_ONLY`：调整并行切分、rank 映射和通信组织；
5. `RUNTIME_ONLY`：调整模型运行时和执行配置；
6. `HYBRID`：组合调整，但需要保留主要变化和归因证据。

每轮闭环为：

```text
模型基线与瓶颈证据
  → 提出模型级假设
  → 生成候选配置或适配补丁
  → 校验权重与派生产物一致性
  → 功能、精度和稳定性门禁
  → 固定拓扑性能测量
  → 接受为模型候选或回退
  → 更新候选谱系与经验
```

对于推理模型，正确性证据可包括输出、任务指标和固定用例；对于训练或微调，可包括损失、梯度、更新行为和收敛契约。具体指标由 ModelSpec 声明，系统不以单一通用阈值替代模型自己的验收标准。

模型级 Campaign 可以创建算子子 Campaign，但必须满足：

- profiling 或覆盖证据已确认算子是功能阻塞点或重要瓶颈；
- 子 Campaign 有独立轮次、时间和资源预算；
- 其他不依赖该算子的模型分析与测试继续并行；
- 返回的算子候选必须重新经过模型级集成验证；
- 单算子最优不自动成为模型级最优。

模型级通过后标记 `MODEL_CANDIDATE_ACCEPTED`。如果用户还要求真实应用或服务收益，该候选继续进入端到端 Campaign。

### 10.6 强化学习双分支与闭环优化

RL 任务可以建立两个并行模型级 Campaign：

- `role=TRAINING`：训练功能、更新行为、收敛契约、训练吞吐、显存和训练通信；
- `role=INFERENCE`：加载与生成、采样语义、权重导入、推理吞吐、时延、显存和推理通信。

两个 Campaign 使用不同基线和 Comparison Contract，可以独立并行推进。任一候选如果修改权重格式、张量布局、采样语义、轨迹字段或版本握手，必须触发新契约版本和对方兼容性验证，不能只在本分支接受。

联合优化时应避免直接枚举全部训练候选与推理候选的笛卡尔积。推荐：

1. 各分支完成功能门禁和独立性能初筛；
2. 选择少量兼容候选组成联合批次；
3. 在限时联合资源窗口中测量权重同步、rollout、轨迹传输和训练更新；
4. 根据闭环主指标接受、回退或重新分配优化预算。

RL 端到端指标可围绕策略迭代时间、有效样本吞吐、样本新鲜度、训练与推理节奏匹配、稳定性或收敛契约定义，由任务选择主指标。训练侧或推理侧的局部提速如果造成另一侧等待增加、权重同步瓶颈、样本过期或闭环稳定性退化，端到端层必须拒绝或回退。

### 10.7 端到端优化闭环

端到端 Campaign 的输入包括：

- 原始基线、模型级候选和单算子候选池；
- 真实模型、框架和部署形态；
- 固定请求、输入或数据分布；
- 固定并发、并行策略和服务配置；
- 端到端 profiling 证据；
- 主指标、正确性与服务约束。

优化动作分为：

1. `KERNEL_ONLY`：替换或继续优化算子；
2. `MODEL_ONLY`：调整模型图、算子组合、并行策略或模型产物；
3. `RUNTIME_ONLY`：调整应用调度、服务并发、显存、通信或部署配置；
4. `HYBRID`：组合调整，但每轮尽量只有一个主要变化，并保留归因证据。

端到端 Campaign 可以：

- 接受微基准最优候选；
- 选择微基准非最优但系统收益更好的候选；
- 因融合、通信重叠或调度效果不佳而回退；
- 判断算子已非瓶颈并停止无效优化；
- 针对已确认瓶颈启动有预算的单算子子 Campaign。

子 Campaign 只返回候选和证据，不自行宣布端到端成功，不允许无限递归。

### 10.8 RSI 与资源利用率协同

单算子 RSI：

- 候选生成和分析不占卡；
- 候选尽量成批生成，构建和测试形成可回填工作包；
- 初筛可并行，复测使用受控环境；
- 每轮结束立即释放卡。

RL 训练与推理分支 RSI：

- 两个分支分别排队和申请资源，可并行初筛和优化；
- 一侧未就绪时不为另一侧的未来联合实验空留资源；
- 最小集成和联合性能测量使用原子联合资源组；
- 联合候选尽量共享一个限时窗口，避免每对候选重复排空；
- 最小闭环结束后释放 SERVICE 租约，恢复双分支独立优化。

模型级、多机或端到端 RSI：

- 在窗口前尽量完成模型转换、候选构建和功能初筛；
- 多个候选共享一个限时独占或静默窗口；
- 窗口内只执行测量所需步骤；
- 窗口结束保存 Campaign checkpoint；
- 窗口外进行分析、候选生成和单卡工作；
- 达到预算、无改善轮次或时间上限后自动停止。

### 10.9 停止条件

- 达到最大轮次或 wall-time；
- 连续若干轮无有效改善；
- 收益低于测量噪声或最小有效收益；
- 正确性无法保持；
- 已确认目标不是端到端瓶颈；
- 资源成本超过用户设定预算；
- 用户终止或改变目标。

## 11. msAgent 接入设计

### 11.1 接入定位

msAgent 作为昇腾领域分析和调优能力提供方接入，不作为第二套资源控制面。其公开项目将自身定位为面向 Ascend NPU 开发、调试和调优的 AI Agent 工作台，组合了 CLI、MCP 工具、Skills 与领域 Agent，覆盖性能、精度、量化和算子等场景。[Ascend/msAgent 项目说明](https://github.com/Ascend/msagent)

本系统优先复用其领域能力，不复制其内部分析流程。接入时需要固定所使用版本和能力清单，避免上游更新导致接口或结果格式漂移。

### 11.2 两种接入方式

#### 方式 A：Skill／MCP 能力级接入

适用于：

- 已采集 profiling 数据的分析；
- 精度问题诊断；
- 算子瓶颈定位；
- 需要把细粒度能力嵌入现有 Task DAG 的场景。

优点是可组合、结果容易标准化、资源占用清晰。公开项目说明也提供了将 Skills 或 msprof-mcp 接入外部 Agent 的路径。

#### 方式 B：完整领域 Agent 调用

适用于：

- 希望复用 msAgent 内部完整工作流；
- 输入输出边界明确、可一次性交付的子任务；
- 不需要本系统介入其每一步决策的场景。

完整 Agent 调用仍必须满足：

- 使用独立工作区；
- 不自行申请或释放本系统的卡；
- NPU 执行由本系统 Runner 完成，或显式传入已有租约上下文；
- 最终返回标准 EvidencePack；
- 超时、取消和恢复由适配器处理。

### 11.3 能力注册表

```yaml
capability_id: msagent.profiler.analysis
provider: msagent
version: ...
kind: DOMAIN_AGENT
inputs:
  - profiling_data
  - question
outputs:
  - analysis_report
  - structured_findings
resource_requirement: NONE
supports_cancel: true
supports_resume: false
timeout_seconds: 1800
```

能力注册表至少描述输入、输出、版本、资源需求、超时、取消和恢复能力。调度器只依据资源需求安排步骤，不依赖能力名称猜测资源。

### 11.4 推荐路由

| 当前需求 | 接入方向 | 典型阶段 |
|---|---|---|
| 模型性能数据归因、算子热点、显存和通信瓶颈分析 | msAgent 性能分析能力 | 模型基线、模型级优化、E2E 基线与优化 |
| 模型或算子精度异常分析 | msAgent 精度分析能力 | 功能验证、分布式验证、集成验证 |
| 算子性能诊断与优化建议 | msAgent 算子调优能力 | 单算子基线与优化 |
| 具体 kernel 修改与实验 | CANNBot Skills 或 Worker | 实现与单算子优化 |
| 资源分配和并行调度 | 本系统 Control Plane | 全阶段 |

msAgent 返回的建议是候选假设，必须经过真实实验后才能成为接受结论。

## 12. CANNBot Skills 接入设计

### 12.1 接入定位

CANNBot Skills 作为可复用的算子开发、调试、验证和优化能力接入。公开项目覆盖多种 DSL 和算子开发流程；本系统通过 Skill Router 按任务类型、DSL 和阶段选择技能，而不是在核心调度器中硬编码技能名称。[CANNBot Skills 项目](https://gitcode.com/cann/cannbot-skills)

### 12.2 Skill Adapter 契约

```yaml
skill_id: cannbot.<domain>.<capability>
version_or_commit: ...
supported_dsl:
  - AscendC
supported_phases:
  - IMPLEMENT
  - OPERATOR_OPTIMIZE
inputs:
  task_description: ...
  workspace: ...
  baseline_evidence_id: ...
  candidate_context: ...
outputs:
  patch: ...
  notes: ...
  required_next_tests: ...
side_effects:
  writes_workspace: true
  executes_npu: false
```

关键要求：

- 明确 Skill 是否会写文件、执行构建或访问 NPU；
- 所有副作用在声明的工作区和租约范围内发生；
- Skill 不自行改变全局任务阶段；
- Skill 不自行启动长期 RSI 循环；
- 输入、输出和版本进入 EvidencePack；
- Skill 不可用时返回明确能力错误，不伪装为实验无结果。

### 12.3 与 RSI 的组合

```text
RSI Leader
  → 形成候选假设
  → Skill Router 选择 CANNBot 能力
  → Candidate Worker 执行单步生成或优化
  → Verifier／Runner 执行真实验证
  → Evidence Registry 保存结果
  → RSI Leader 比较并决定下一轮
```

基线冻结、候选目录、实验预算、迭代、比较和回退由外层 Campaign 管理；Skill 聚焦领域单步能力。这样既能替换 Skill 版本，也能替换 RSI 实现而不互相侵入。

### 12.4 结果归一化

无论结果来自 msAgent、CANNBot Skills 还是普通 Worker，都转换为：

```text
DomainResult
  ├── status
  ├── capability_id / version
  ├── patch_digest
  ├── findings[]
  ├── proposed_hypotheses[]
  ├── required_tests[]
  ├── evidence_ids[]
  └── limitations[]
```

`proposed_hypotheses` 不自动等于 `accepted_candidates`；`findings` 不自动等于事实门禁，必须关联原始证据。

## 13. Kimi Code 嵌入方式

### 13.1 组件形态

建议封装为项目级或用户级插件：

```text
ascend-orchestration-plugin/
├── agents/
│   ├── ascend-orchestrator.md
│   ├── ascend-model-worker.md
│   ├── ascend-single-worker.md
│   └── ascend-distributed-owner.md
├── skills/
│   ├── task-phase-policy/
│   ├── model-adaptation-policy/
│   ├── operator-campaign-policy/
│   ├── model-campaign-policy/
│   └── e2e-campaign-policy/
├── adapters/
│   ├── msagent/
│   └── cannbot/
├── hooks/
│   └── guard-agent-mode
└── mcp configuration
```

Kimi Code 自定义 Agent 可声明工具白名单和子 Agent 白名单，子 Agent 具有独立上下文；MCP 用于连接外部控制面和领域工具。[Kimi Code Agents 文档](https://www.kimi.com/code/docs/en/kimi-code-cli/customization/agents.html)、[Kimi Code MCP 文档](https://www.kimi.com/code/docs/en/kimi-code-cli/customization/mcp.html)

### 13.2 Agent 权限

- Orchestrator：可委派 Model Worker 和 Single Worker，可调用阶段、Campaign 和资源 MCP，不直接访问 NPU shell。
- Model Worker：处理模型适配工作包，只能使用声明的模型产物和独立工作区，无继续委派和直接 NPU shell 权限。
- Single Worker：独立工作区，无继续委派权限，无直接 NPU shell。
- Distributed Owner：无执行型子 Agent，统一操作一个分布式任务。
- RSI Leader：可生成候选和调用领域适配器，不直接分配物理资源。
- 只读分析 Agent：可在不违反任务 Owner 约束的情况下处理离线证据，但不能执行或修改分布式现场。

### 13.3 Hook 的定位

Hook 可用于：

- 在独占阶段提醒或拦截与独占资源集合冲突的新执行型 Agent 委派；
- 对危险 shell 命令进行辅助拦截；
- 在子 Agent 完成、任务失败和窗口到期时发送通知；
- 向上下文追加当前任务阶段和资源状态。

Hook 不能作为唯一安全屏障，因为其错误处理可能放行。最终安全由 Runner、设备权限、租约校验和围栏机制保证。[Kimi Code Hooks 文档](https://www.kimi.com/code/docs/en/kimi-code-cli/customization/hooks.html)

## 14. 控制面接口设计

### 14.1 任务与阶段

| 接口 | 说明 |
|---|---|
| `task_create` | 创建任务轨道、目标、验收条件、预算和初始 DAG |
| `model_spec_register` | 注册模型源产物、工作负载、目标拓扑和验收契约 |
| `model_artifact_validate` | 校验源模型、派生产物、配置、分片和跨节点一致性 |
| `rl_contract_freeze` | 冻结 RL 权重、轨迹、采样和版本握手契约，解除双分支并行依赖 |
| `rl_contract_update` | 创建新契约版本并标记受影响步骤，禁止静默变更 |
| `rl_integration_validate` | 校验双方最小集成证据、契约版本和产物兼容性 |
| `task_get` | 获取阶段、步骤、依赖、活动和历史证据 |
| `task_update_dag` | 新增或修改工作包，检查所有权与依赖 |
| `phase_transition` | AI 提交阶段转换和证据 |
| `task_pause` | 停止新步骤准入，保留现场 |
| `task_resume` | 恢复 DAG 和调度 |
| `task_cancel` | 受控停止、清理和归档 |

### 14.2 资源与租约

| 接口 | 说明 |
|---|---|
| `cluster_status` | 动态节点、卡、标签、资源版本、构建槽、队列、预留和健康状态 |
| `inventory_watch` | 订阅节点加入、退出、cordon、drain、健康与卡清单变化 |
| `resource_group_acquire` | 为 RL 闭环等联合场景原子申请多角色资源 |
| `resource_group_release` | 统一停止联合角色，确认清理后释放全部资源 |
| `lease_acquire` | 为 READY 步骤原子申请资源 |
| `lease_heartbeat` | 续租并报告进度 |
| `lease_release` | 结束执行并进入清理，确认安全后释放可分配资源并重新调度 |
| `reservation_request` | 为已就绪的大任务申请未来窗口 |
| `reservation_status` | 查询排空、回填和阻塞情况 |
| `reservation_cancel` | 取消未生效预留并释放保护资源 |

### 14.3 执行

| 接口 | 说明 |
|---|---|
| `run_build` | 使用 CPU 构建槽执行受控构建 |
| `run_single_card_test` | 使用单卡租约执行功能或性能测试 |
| `run_single_node_test` | 使用单机资源集合执行多 rank 任务 |
| `run_multi_node_test` | 使用多机资源集合执行分布式任务 |
| `run_rl_integration` | 在联合资源组中执行受控权重交接、轨迹交接和 rollout—train 闭环 |
| `run_profile` | 在声明的隔离范围内采集 profiling |
| `run_cleanup` | 按步骤和资源集合执行清理 |
| `execution_status` | 获取状态、日志位置和退出原因 |

普通执行接口均要求 `task_id`、`step_id`、`lease_id` 和 `epoch`；联合执行还要求 `resource_group_id`，并校验组内全部角色租约。所有执行接口返回标准执行记录。

### 14.4 Campaign 与领域能力

| 接口 | 说明 |
|---|---|
| `campaign_create` | 创建基线、Comparison Contract 和预算 |
| `campaign_append_candidate` | 注册候选、父候选和假设 |
| `campaign_record_evaluation` | 写入真实评估和决策 |
| `campaign_checkpoint` | 保存可恢复状态 |
| `campaign_stop` | 记录停止原因并释放窗口需求 |
| `capability_list` | 查询 msAgent、CANNBot 等已注册能力 |
| `capability_invoke` | 通过适配器调用领域能力 |
| `capability_cancel` | 取消能力调用并回收相关步骤 |

### 14.5 幂等与审计

所有写接口接受 `request_id`。重复请求返回原结果，避免网络重试造成重复租约、重复运行或重复候选。每次状态变化追加事件日志，至少包括操作者、时间、输入摘要、前后状态和原因。

## 15. 持久化、环境指纹与产物管理

### 15.1 元数据表

建议使用支持事务的持久化数据库。单控制节点原型可使用 SQLite，正式多服务部署再迁移到服务端数据库。

核心表：

```text
tasks
model_specs
model_artifacts
rl_specs
resource_groups
steps
step_dependencies
resource_inventory
leases
reservations
executions
campaigns
candidates
comparison_contracts
evidence
capabilities
events
```

租约和资源分配必须事务化；事件日志只追加，不覆盖历史调度原因。

### 15.2 环境指纹

每次真实执行至少记录：

- 节点和硬件清单摘要；
- 驱动、固件和 CANN 运行时版本；
- 框架与关键依赖版本；
- 仓库 commit、补丁摘要和构建参数；
- 算子二进制或产物摘要；
- 模型源权重、配置、分片、预处理器和派生产物摘要（模型任务必填）；
- 模型并行策略、rank 拓扑和关键运行时配置（模型任务必填）；
- RL 契约版本、训练／推理角色、权重版本、轨迹 schema 和采样配置（RL 任务必填）；
- 测试入口、配置摘要和随机种子；
- 资源拓扑与隔离等级；
- 测量协议摘要。

版本字段来自实际环境采集，不在设计中硬编码。

### 15.3 产物管理

- 构建产物按内容摘要寻址；
- 多机运行前验证各节点产物摘要一致；
- 日志和性能原始数据与 EvidencePack 关联；
- 失败候选同样保留，不覆盖为后续成功结果；
- 清理运行进程不等于删除证据；
- 对日志中的凭据、路径和敏感信息进行访问控制与必要脱敏。

## 16. 可观测性与效率指标

### 16.1 资源指标

**健康卡利用率：**

```text
U_npu = 设备实际执行任务卡时 / 健康可分配卡时
```

**有效卡时占比：**

```text
U_effective = 产生有效证据的必要执行卡时 / 健康可分配卡时
```

**可避免空闲卡时：**

```text
存在兼容 READY 步骤、无互斥和预留限制，但仍未分配的空闲卡时
```

同时记录：

- 预留保护空闲卡时；
- 静默窗口空闲卡时；
- 故障和清理占用卡时；
- 资源碎片化导致的等待时间；
- 构建槽利用率和构建排队时间；
- 动态节点从加入到可分配的耗时、资源视图版本滞后和错误拒绝分配次数。

### 16.2 任务指标

- READY 到 RUNNING 的排队时间分布；
- 分布式任务从申请到资源齐备的排空时间；
- 每个 Task 的端到端完成时间；
- 模型任务从创建到最小可行运行、整模型功能通过和目标分布式拓扑通过的分别耗时；
- 模型算子覆盖率、缺失算子子 DAG 完成时间和模型产物复用率；
- 权重转换、加载、分发和图编译占用的等待时间；
- RL 训练与推理分支各自达到最小可行运行和可集成条件的耗时；
- RL 因等待另一分支、权重同步、轨迹传输和联合资源组产生的等待时间；
- 抢占次数、重跑次数和切换开销；
- 等待老化触发次数和最长等待时间；
- 单位时间成功步骤数；
- 主 Agent 等待子 Agent 结果的整合时间。

### 16.3 RSI 指标

- 每小时真实完成的有效候选评估数；
- 正确性门禁通过率；
- 测量无效率与复测率；
- 接受候选数及累计回退次数；
- 单算子候选在模型级和端到端阶段的保留率；
- 模型级候选在端到端阶段的保留率；
- 每个有效收益所消耗的卡时、模型转换工时和独占窗口时长；
- 达到无改善停止条件前的轮次。

### 16.4 正确性指标

以下指标应始终为零：

- 同卡冲突分配；
- 一个分布式任务出现多个执行 Owner；
- 过期租约成功执行；
- 多机节点产物摘要不一致仍启动；
- 模型权重、配置、分片或派生产物不一致仍启动跨机运行；
- RL 契约或权重版本不兼容仍启动联合闭环；
- 联合资源组只分配部分角色就启动闭环；
- 正确性失败候选被性能 Campaign 接受；
- 清理状态未确认就重新分配资源。

高利用率不能抵消上述任何一项正确性事故。

## 17. 安全与权限边界

1. 资源控制面和 Runner 失败关闭，Hook 仅做辅助提醒与拦截。
2. Agent 工作区不默认暴露直接 NPU 设备执行能力。
3. 所有第三方 Agent、MCP 和 Skill 仅在可信环境中启用。
4. msAgent、CANNBot Skills 的版本、来源和能力清单进入审计记录。
5. 领域 Agent 如需 shell 或外部工具，其权限由适配器和运行沙箱收敛。
6. 多机启动、全局静默、强制抢占和大范围清理由策略控制；高风险操作可配置人工审批。
7. 普通自动阶段切换不需要用户逐次确认，但所有转换必须可见、可解释、可暂停。
8. 凭据通过受控环境注入，不写入候选、日志或 EvidencePack。
9. 动态节点加入必须经过 Executor 身份认证、资源池准入和健康检查，动态发现不等于允许任意节点自行进入调度池。
10. RL 权重交换和轨迹交换使用受控通道，契约版本、来源摘要和接收校验结果进入审计记录。

## 18. 推荐初始配置

```yaml
cluster:
  discovery:
    enabled: true
    seed_executors: []            # 可配置初始 Executor，实际节点和卡以注册上报为准
    accept_resource_pools:
      - default
  require_ready_before_allocation: true
  stale_inventory_rejects_allocation: true
  fixed_total_cards: null

tasks:
  enabled_tracks:
    - OPERATOR_DEVELOPMENT
    - MODEL_ADAPTATION
    - MIXED
  model_source_artifacts_readonly: true
  model_minimal_viable_run_first: true
  allow_operator_subdag: true
  rl:
    freeze_contract_before_branches: true
    training_inference_parallel: true
    integration_waits_for_full_optimization: false
    atomic_resource_group: true

agents:
  max_total_workers: 8
  max_single_workers: 8
  max_model_workers: 4
  max_distributed_owners_per_task: 1
  max_rl_integration_owners_per_task: 1
  allow_worker_delegation: false

execution:
  build_slots_per_node: 2
  lease_heartbeat_seconds: 20
  lease_timeout_seconds: 60
  default_step_timeout_seconds: 1800
  service_lease_max_duration_seconds: 7200
  service_lease_renewable: true
  direct_npu_shell: false

scheduler:
  policy: WORK_CONSERVING
  backfill: ENABLED
  backfill_duration_percentile: P90
  starvation_aging: ENABLED
  static_partition: false
  reservation_requires_ready_task: true
  preemption: CHECKPOINT_FIRST

distributed:
  default_exclusion_scope: TARGET_RESOURCE_SET
  functional_debug_quiet: NONE
  performance_default_quiet: NODE_SET_QUIET
  global_quiet_requires_explicit_evidence: true
  lease_upgrade_in_place: false

rsi:
  single_leader_per_campaign: true
  candidate_generation_uses_npu: false
  batch_distributed_evaluations: true
  operator_result_is_e2e_acceptance: false
  nested_campaign_requires_budget: true

adapters:
  pin_versions: true
  normalize_evidence: true
  allow_independent_resource_control: false
```

`performance_default_quiet=NODE_SET_QUIET` 是保守起点。如果测量证明参与节点内部的其他任务不影响结果，可缩小到设备独占；如果共享网络或跨节点任务仍造成不可接受波动，再扩大为 `GLOBAL_QUIET`。

## 19. 部署形态

### 19.1 控制面

- 管理节点部署 Controller、数据库和 Evidence Registry；
- 通过 HTTP MCP 向 Kimi Code 暴露受控工具；
- 通过内部 API 与所有已注册 Node Executor 通信，并处理节点动态加入、退出和资源版本变化；
- 所有写操作幂等，关键状态事务化；
- Controller 重启后与 Executor 对账，不直接假设旧任务仍运行。

### 19.2 执行面

每个 Node Executor 负责：

- 动态注册节点，上报实际卡清单、健康卡、硬件与运行时标签、构建槽和环境指纹；
- 处理 cordon、drain、资源版本变化和重新加入前的现场检查；
- 创建和回收工作区；
- 按租约注入设备可见性；
- 启动受控构建、单卡、分布式或 RL 联合角色进程；
- 收集退出码、日志和性能数据；
- 维护任务进程组并执行精确清理；
- 响应取消、检查点和围栏撤销。

### 19.3 代码与产物

- 每个候选使用独立 worktree；
- 共享仓库和缓存必须支持并发；
- 构建产物通过内容寻址存储分发；
- 模型源权重、配置和预处理器只读共享，派生权重、图、引擎和切分配置按源摘要与转换链路寻址；
- 多机执行采用统一运行清单、模型产物摘要和算子产物摘要；
- 日志按 Task、Step、Candidate、Model Artifact、rank 分层组织。

## 20. 分阶段实施计划

### 阶段一：自动阶段决策与高利用率资源闭环

交付内容：

1. 动态 Executor 注册、实际节点／卡清单、标签、健康检查、资源版本和节点上下线；
2. Task、Step DAG、ResourceSpec、Lease 和 EvidencePack；
3. ModelSpec、模型源产物与派生产物注册、摘要和只读共享；
4. 主 Agent 自动识别模型、RL、算子或混合任务轨道并判断阶段，不提供常规手动模式入口；
5. 模型最小可行运行、算子覆盖检查和缺失算子子 DAG；
6. RLSpec、交互契约冻结、训练与推理双分支 DAG、分支独立功能验收；
7. RL 最小权重交接、轨迹交接、联合资源组和 rollout—train 闭环；
8. 单卡工作包并行、独立工作区和按步骤租约；
9. 单机、多机通信、分布式模型和 RL 闭环任务单 Owner；
10. 目标资源集合排空，非冲突资源继续调度；
11. 基础工作保持队列和保守回填；
12. 租约心跳、纪元围栏、SERVICE 租约、超时清理和 Controller 恢复；
13. msAgent、CANNBot 能力注册表和适配器契约；
14. 至少各打通一个离线能力调用或受控 Skill 调用示例。

阶段一不实现复杂自动候选优化，但所有数据模型保留 Campaign 扩展字段。

### 阶段二：单算子 RSI

1. RSI Leader 和 Campaign 持久化；
2. Comparison Contract、冻结基线和正确性门禁；
3. 单卡候选并行初筛与受控复测；
4. 失败候选留痕、接受、回退和 checkpoint；
5. CANNBot Skills 作为候选生成、调试或验证能力接入；
6. msAgent 作为 profiling 分析和瓶颈假设来源；
7. Campaign 预算和无改善停止。

### 阶段三：模型级与端到端 RSI

1. 模型级基线：绑定模型源产物、派生产物、工作负载、固定拓扑和模型验收契约；
2. 模型级动作空间：算子组合、图、显存、并行、模型运行时和受限混合候选；
3. 模型级功能、精度、稳定性与性能门禁，支持推理和训练各自的可配置证据；
4. RL 训练与推理独立 Campaign、契约兼容性门禁和并行候选初筛；
5. RL 联合候选批次、权重同步、轨迹传输和 rollout—train 闭环主指标；
6. 真实应用负载基线和端到端 EvidencePack；
7. `KERNEL_ONLY`、`MODEL_ONLY`、`RUNTIME_ONLY` 和受限 `HYBRID` 动作；
8. 单算子与模型级候选池、模型级接受和端到端最终接受；
9. 限时节点集合、联合资源组或全局静默窗口；
10. 多候选批量测量和窗口外分析；
11. 有预算的瓶颈算子、训练分支、推理分支或模型级子 Campaign，禁止无预算递归；
12. 模型级、RL 分支和端到端退化自动回退。

### 阶段四：调度优化与规模化

1. 基于历史时长的智能回填；
2. 拓扑和通信干扰画像；
3. 动态调整 Agent、构建槽和候选批量；
4. 更精细的关键路径、优先级和公平性策略；
5. 多 Campaign 资源预算和利用率看板；
6. 根据测量数据调整节点集合与全局静默策略。

## 21. 验收测试

### 21.1 基础调度与动态集群

1. **单卡填满测试**：提交超过当前健康卡总数的单卡测试，验证无重复分配，Agent 上限与设备队列相互独立。
2. **工作保持测试**：存在兼容 READY 步骤且无预留限制时，健康空闲卡必须被分配或给出明确原因。
3. **思考不占卡测试**：Worker 在编码和分析期间不持有 NPU 租约。
4. **构建隔离测试**：并发构建不覆盖目录，不超过每机构建槽。
5. **依赖测试**：依赖未满足的 Step 不得进入 RUNNING。
6. **节点动态加入测试**：新 Executor 注册并通过健康检查后，其资源自动进入调度池，无需修改任务模板或重启 Controller。
7. **节点 cordon／drain 测试**：cordon 后无新分配，已有任务按策略结束；drain 后冲突任务安全排空，节点恢复前不复用旧租约。
8. **资源版本与异构选择测试**：调度匹配后资源版本变化时重新校验；标签不满足的节点即使空闲也不能分配。

### 21.2 模式切换

9. **自动升级测试**：AI 根据测试动作从单卡功能验证进入单机或多机，无需用户输入模式命令。
10. **部分节点联调测试**：目标节点集合排空时，非冲突节点继续运行独立任务。
11. **全资源联调测试**：请求覆盖当前全部可用资源时，所有冲突任务安全结束或受控抢占，分布式任务只有一个 Owner。
12. **回填测试**：预留开始前，满足 P90 时长和清理余量的短任务可以执行；超时任务不能进入。
13. **降级测试**：多机问题缩小为单卡问题后，独占资源释放并恢复并行。
14. **防饿死测试**：持续提交单卡任务时，多机任务通过老化和预留最终获得资源。

### 21.3 模型适配与混合任务

15. **模型最小资源路径测试**：最终目标为多机的模型，在静态分析、配置转换和最小功能阶段不得提前独占多机。
16. **缺失算子子 DAG 测试**：覆盖检查发现缺失算子后自动创建算子子任务，不依赖该算子的模型分析和测试继续并行。
17. **分层验收测试**：模块运行、整模型运行、单卡精度、分布式功能和性能证据不得相互替代。
18. **模型产物一致性测试**：权重、配置、分片或派生产物摘要不一致时，跨机运行必须被阻止。
19. **派生产物复用测试**：源产物与转换链路一致时可安全复用；源模型或转换参数变化后，旧产物不能继续冒充同一基线。
20. **分布式模型 Owner 测试**：同一分布式模型任务只有一个执行 Owner，问题缩小后能够降级并释放资源。

### 21.4 强化学习双分支与闭环

21. **RL 双分支并行测试**：契约冻结后，训练与推理步骤可同时进入 READY，并根据各自最小拓扑独立获得资源。
22. **单侧阻塞隔离测试**：训练分支等待算子或数据时，推理分支继续推进，反之亦然；不得为未就绪闭环提前空留资源。
23. **最小集成依赖测试**：双方达到最小可集成条件即可进入 RL 集成，不要求任何一侧先完成全部性能优化。
24. **契约版本测试**：权重格式、张量布局、采样语义或轨迹字段变化必须生成新版本，版本不兼容时禁止联合运行。
25. **联合资源原子性测试**：训练和推理资源不能同时满足时，不得启动半组闭环；超时后整体回滚或继续明确排队，不能形成双向持有死锁。
26. **RL 闭环证据测试**：至少完成训练导出、推理导入、轨迹生成、训练消费、新权重回同步和最小多轮 rollout—train，分支单独通过不能替代该证据。

### 21.5 故障与安全

27. **旧租约围栏测试**：租约过期并重新分配后，旧 Worker 的执行必须被拒绝。
28. **Controller 重启测试**：恢复后与 Executor 对账，不重复分配、不重复执行幂等请求。
29. **Executor 失联测试**：节点停止新分配，恢复后先清理再上线。
30. **部分 rank 失败测试**：保存现场，整个运行按失败处理，由唯一 Owner 决定后续动作。
31. **Hook 失效测试**：即使辅助 Hook 未执行，Runner 仍阻止未授权 NPU 操作。

### 21.6 RSI 与适配器

32. **正确性门禁测试**：性能更快但正确性失败的候选不得接受。
33. **基线防漂移测试**：基线代码、模型产物、协议或环境变化必须生成新基线，不能混用结果。
34. **模型级接受测试**：算子候选只有重新通过模型集成功能、精度和固定拓扑性能验证后，才能成为模型级候选。
35. **端到端否决测试**：单算子或模型级提速但端到端退化时，最终阶段必须回退。
36. **RL 分支优化否决测试**：训练或推理局部提速如果导致另一侧等待、样本过期、权重同步瓶颈或闭环退化，联合层必须拒绝或回退。
37. **RL 候选组合控制测试**：联合评估先经过独立初筛和契约兼容性过滤，不直接无预算枚举两侧候选笛卡尔积。
38. **Campaign 恢复测试**：窗口结束或服务重启后可从 checkpoint 继续，不重复扣减或丢失候选。
39. **单 Leader 测试**：同一 Campaign 不能出现第二个独立接受候选的 Leader。
40. **msAgent 适配器测试**：能力超时或失败被明确记录，不被解释为“无问题”。
41. **CANNBot 适配器测试**：Skill 版本、补丁和所需后续测试完整进入证据包。
42. **性能静默测试**：静默窗口内无未授权干扰任务，窗口外资源立即恢复工作保持调度。

## 22. 风险与应对

| 风险 | 表现 | 应对 |
|---|---|---|
| AI 阶段误判 | 资源申请过大或执行接口选错 | 最小资源原则、证据门禁、Runner 强校验 |
| 模型任务过早扩大拓扑 | 尚未完成最小运行就长期占用多机 | 模型专用阶段模板、最小可行运行门禁 |
| 模型产物污染或错用 | 候选共享可写权重、跨机分片不一致 | 源产物只读、派生产物隔离、摘要一致性校验 |
| 模型与算子任务割裂 | 缺失算子阻塞整个模型，或算子收益未回到模型验证 | 混合 Task DAG、算子子 Campaign、模型级复验 |
| 固定集群假设失效 | 节点扩缩容后仍按旧卡数或旧节点分配 | 动态发现、资源版本、标签选择、分配前复核 |
| RL 双分支被串行化 | 一侧未开始或长期等待另一侧全部完成 | 契约先行、独立 DAG、最小集成依赖 |
| RL 契约漂移 | 训练和推理各自通过但无法交换权重或轨迹 | 版本化 RLSpec、兼容性门禁、联合闭环证据 |
| 联合资源死锁 | 训练和推理各持有部分资源并相互等待 | 原子 ResourceGroup、超时整体回滚 |
| 大任务长期排不到 | 单卡任务持续进入目标节点 | 就绪后预留、老化、回填截止时间 |
| 为预留造成空置 | 大任务未就绪就清空资源 | 仅就绪任务可预留、工作保持、受控回填 |
| 过度并行导致抖动 | 构建、存储和主 Agent 整合拥塞 | Agent 与卡并发分离、构建槽、切换成本评分 |
| 回填估计错误 | 预留延迟或任务频繁取消 | 历史 P90、保守余量、检查点优先、记录误差 |
| 性能测量互相干扰 | 结果波动、错误接受候选 | 节点集合或全局静默、复测和异常标记 |
| RSI 长期占集群 | 探索任务阻塞功能开发 | 预算、限时窗口、批量测量、优先级 |
| 微基准收益无法转化 | 单算子快但端到端退化 | 候选池与端到端最终接受分离 |
| 外部能力版本漂移 | 同一输入结果不可比 | 版本固定、能力注册、环境指纹 |
| 多套控制面冲突 | msAgent 或其他 Agent 自行抢资源 | 适配器禁止独立资源控制、统一 Runner |
| 清理不彻底 | 残留 rank、端口或设备占用 | 资源未确认安全前不重新分配 |

## 23. 关键设计决策汇总

1. 模型适配与算子工程都是一等任务轨道，RL 是模型适配的专用场景模板，混合任务通过统一 Task DAG 编排，不建设多套调度系统。
2. 集群不固定为四台或 64 卡；Executor 动态注册实际节点、卡、标签和拓扑，节点上下线通过资源版本参与调度。
3. 不做静态卡分区，所有当前健康且兼容的资源统一进入动态资源池。
4. 不使用 `/single-card`、`/single-node`、`/multi-node` 作为常规入口，由 AI 按任务轨道、阶段和下一动作自动决策。
5. 任务阶段与资源请求正交，按 Step 调度、按实际执行持卡；模型最终需要多机不代表适配初期就占用多机。
6. 模型适配以最小可行运行、算子覆盖、功能精度、分布式验证和模型级性能形成分层证据，禁止用局部运行替代整模型验收。
7. RL 先冻结权重交换、轨迹、采样和版本握手契约，训练与推理再作为独立子 DAG 并行适配；任一分支阻塞不应暂停另一分支。
8. RL 最小集成只依赖双方达到可集成条件，不等待全部性能优化；最终必须补充权重交接、轨迹交接和 rollout—train 闭环证据。
9. 模型源权重与配置只读共享，派生产物按内容摘要管理；跨机和 RL 联合运行前校验产物与契约版本一致性。
10. RL 联合在线角色使用原子 ResourceGroup 和有期限的 SERVICE 租约，不为未来闭环提前空留资源，防止部分分配死锁。
11. 同一个通信、分布式模型或 RL 闭环任务只有一个 Owner，但非冲突节点和其他独立分支可以继续运行。
12. 调度采用工作保持、受控回填、大任务预留和等待老化，避免“填满”与“排空”两个极端。
13. Agent 并发、NPU 并发、构建并发和模型转换／分发开销分别治理，Agent 数量不直接从总卡数推导。
14. 所有 NPU 操作经过统一 Runner，租约、纪元和设备权限形成最终安全边界。
15. RSI 位于功能正确性之后，分为单算子、模型级和端到端三层；RL 训练与推理可分别运行模型级 Campaign，联合层拥有闭环接受权。
16. 每个 Campaign 只有一个 Leader，外部 Skill 不独立维护第二套优化循环，子 Campaign 必须有预算。
17. msAgent 和 CANNBot Skills 通过能力注册表、适配器和 EvidencePack 接入，不侵入调度核心。
18. 多机及 RL 联合性能候选采用限时批量实验窗口，窗口外进行不占卡的生成、转换和分析。
19. 资源利用率与有效研发产出同时度量，任何利用率目标都不能突破正确性、模型／RL 验收和可复现性门禁。

## 24. 参考资料

- [Ascend/msAgent：面向昇腾开发、调试和调优的 AI Agent 工作台