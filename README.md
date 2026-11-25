# Ascend Orchestrator Prototype

昇腾模型与算子适配智能体的资源调度控制面原型，实现设计文档 V1.2 的**阶段一：自动阶段决策与高利用率资源闭环**。

项目使用 Python 标准库和 SQLite，无需安装运行时依赖，适合本地验证调度语义、Task DAG、模型产物契约和 RL 双分支编排。

> **原型边界**：当前 Runner 是模拟执行器，不访问 NPU，也不构成真实硬件精度或性能验收。生产部署需要接入远程 Node Executor、认证、设备权限和进程清理机制。

## 功能概览

- **动态集群**：Executor 注册与心跳、实际卡清单、标签、资源版本、cordon/drain/lost 和恢复对账。
- **统一任务编排**：Task/Step DAG、依赖门禁、自动轨道识别、模型最小资源路径和缺失算子子 DAG。
- **确定性调度**：单卡并行、单机/多机拓扑匹配、工作保持、等待老化、预留和保守回填。
- **执行隔离**：单 Owner、独立运行目录、设备/节点/静默互斥、按步骤租约和纪元围栏。
- **模型契约**：ModelSpec、只读源摘要、派生产物复用与漂移检查、多机产物一致性门禁。
- **RL 双分支**：契约冻结与升级、训练/推理独立 DAG、原子 ResourceGroup 和最小闭环证据。
- **可审计扩展**：EvidencePack、追加式事件日志、幂等写请求、msAgent/CANNBot 适配器示例。

Campaign/Candidate 数据表已预留，阶段一不包含自动候选优化。

## 环境要求

- Python 3.9 或更高版本
- macOS、Linux 或其他支持 SQLite 的 Python 环境
- 无第三方运行时依赖，无需 NPU 即可运行演示和测试

## 快速开始

### 直接运行

```bash
git clone https://github.com/lifenghit/ascend_orchestrator.git
cd ascend_orchestrator

# 运行模拟演示；重复执行时使用 --reset 替换演示数据库
PYTHONPATH=src python3 -m ascend_orchestrator demo --reset

# 查看集群与队列状态
PYTHONPATH=src python3 -m ascend_orchestrator --db var/demo.db status

# 查看审计事件
PYTHONPATH=src python3 -m ascend_orchestrator --db var/demo.db events
```

演示会创建两个模拟节点，依次完成算子任务、RL 训练/推理独立适配和原子联合闭环。输出中的任务状态、Evidence ID 和资源释放结果均可通过 SQLite 状态继续查询。

### 安装命令行工具

建议使用虚拟环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip setuptools
python3 -m pip install -e .

ascend-orchestrator demo --reset
ascend-orchestrator --help
```

Windows 激活虚拟环境的命令为 `.venv\Scripts\activate`。

## CLI

全局参数必须放在子命令之前：

```text
ascend-orchestrator [--db PATH] [--workspace PATH] COMMAND
```

| 命令 | 说明 |
| --- | --- |
| `demo [--reset]` | 运行模拟算子与 RL 闭环演示；默认不覆盖已有数据库 |
| `serve [--host HOST] [--port PORT]` | 启动本地 HTTP JSON API，默认 `127.0.0.1:8765` |
| `status` | 输出节点、健康卡、队列、预留和可避免空闲状态 |
| `events` | 输出追加式审计事件 |

默认数据库为 `ascend.db`，工作区为 `var/workspaces`。演示默认使用 `var/demo.db` 和 `var/demo-workspaces`。

## HTTP API

```bash
PYTHONPATH=src python3 -m ascend_orchestrator serve
```

另一个终端中执行：

```bash
curl -s http://127.0.0.1:8765/health

curl -s -X POST http://127.0.0.1:8765/executors/register \
  -H 'Content-Type: application/json' \
  -d '{"node_id":"node-0","npus":[{"id":0},{"id":1}],"labels":{"runtime_profile":"a"},"request_id":"register-node-0-v1"}'

curl -s -X POST http://127.0.0.1:8765/tasks \
  -H 'Content-Type: application/json' \
  -d '{"objective":"开发并验证 AscendC 算子","request_id":"create-operator-task-v1"}'

curl -s -X POST http://127.0.0.1:8765/schedule
curl -s http://127.0.0.1:8765/cluster
```

完整接口、错误语义和 RL 示例见 [HTTP API 使用说明](docs/api.md)。

## 测试

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

测试覆盖动态资源、标签选择、构建槽、依赖、回填、模型产物、RL 原子分配与契约版本、闭环证据、单 Owner、租约围栏、恢复、静默窗口和适配器。

## 项目结构

```text
.
├── src/ascend_orchestrator/
│   ├── models.py        # ResourceSpec、Task、Step、Lease、EvidencePack 合同
│   ├── store.py         # SQLite 事务、幂等和事件日志
│   ├── scheduler.py     # 工作保持、拓扑、互斥、回填和原子资源组
│   ├── orchestrator.py  # 自动轨道识别、DAG 模板和阶段推导
│   ├── control.py       # 控制面、执行门禁、模型与 RL 流程
│   ├── adapters.py      # msAgent / CANNBot 契约示例
│   ├── api.py           # HTTP JSON API
│   ├── demo.py          # 可重复演示
│   └── cli.py           # 命令行入口
├── tests/               # 阶段一核心验收测试
├── docs/                # 原始设计与接口文档
└── .github/workflows/   # 持续集成
```

## 文档与后续阶段

- [系统设计 V1.2](docs/design-v1.2.md)
- [HTTP API 使用说明](docs/api.md)
- [贡献与开发指南](CONTRIBUTING.md)

后续阶段依次为单算子 RSI、模型级与端到端 RSI、调度优化与规模化。当前调度采用确定性在线启发式，不追求全局最优；P90 历史估计、复杂公平性和利用率看板不属于本原型。

## 许可证

本项目采用 [MIT License](LICENSE)。允许使用、复制、修改、合并、发布、分发、再许可和商用，但需保留版权与许可声明。软件按“原样”提供，不附带任何担保。
