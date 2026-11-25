# 贡献与开发指南

## 本地开发

项目不要求第三方运行时依赖。推荐使用 Python 3.9 或更高版本：

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip setuptools
python3 -m pip install -e .
python -m unittest discover -s tests -v
ascend-orchestrator demo
```

不安装项目时，可在命令前设置 `PYTHONPATH=src`。

## 修改原则

- 资源分配、租约状态变更和联合资源组分配必须保持事务性。
- 所有写接口必须支持 `request_id` 幂等，并记录追加式审计事件。
- AI 或领域适配器负责提出需求，调度器负责硬约束；不要在 Skill 中建立第二套资源控制面。
- 分析、编码和候选生成不得隐式持有 NPU 租约。
- 不得用模拟 Runner 的证据宣称真实 NPU 精度或性能通过。
- 新增公开字段应保持向后兼容；Campaign/Candidate 扩展不能破坏阶段一数据模型。
- 修改用户可见行为时同步更新 `README.md`、`docs/api.md` 或相关设计说明。

## 测试要求

提交前至少运行：

```bash
python -m compileall -q src tests
python -m unittest discover -s tests -v
ascend-orchestrator demo --reset
```

测试数据库和工作区应使用临时目录，不要提交 `var/`、SQLite 数据库、日志或运行证据。

涉及调度、RL、模型产物或租约的修改，应添加正反向用例。例如联合资源组既要验证全部分配成功，也要验证资源不足时没有任何角色占用资源。

## 提交与 Pull Request

- 一个 PR 聚焦一个可审查的主题。
- 说明行为变化、验证命令和结果，以及仍未覆盖的真实硬件边界。
- 不提交凭据、私有模型权重、内部路径或包含敏感信息的日志。
- CI 在 Python 3.9 和 3.12 上运行测试及端到端模拟演示，需全部通过。
