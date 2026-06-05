# Qwen / Claude 人工 Rubrics 评分模板说明

这是 10 条 Coding Trajectory 投标样例的正式人工评分模板。

当前模板按 `docs/examples/` 中的样例文件口径整理：

- `qwen/` 下 10 个 Qwen 评分文件。
- `claude/` 下 10 个 Claude 评分文件。
- 每个文件 7 个评分维度。
- 每个维度 `points = 5`。
- 每个维度 `weight = 1.0`。
- `passrate = score 总和 / 35`。

## 字段从哪里来

- `name`：评分维度的英文标识，依据需求文档中的质检原则和 rubrics 要求整理。
- `description`：针对当前任务写出的评分说明，来自这 10 条任务的需求、验收点和风险点。
- `type`：固定为 `likert`，表示人工按 0-5 分打分。
- `points`：固定为 `5`，与官方示例一致。
- `weight`：固定为 `1.0`，与官方示例一致。
- `score`：你看完 trajectory 和最终代码结果后人工填写。
- `rationale`：你人工填写评分理由，要写具体证据。

这些 `name` 和 `description` 不是 Claude 或 Qwen 在跑题时直接生成的，而是根据任务要求和需求文档里的质检口径提前设计好的评分模板。Qwen / Claude 只产生 trajectory 和最终结果；评分文件由你人工填写。

## 填写规则

- 只填写 `score` 和 `rationale`。
- 不建议修改 `name`、`description`、`type`、`points`、`weight`。
- 每个 `score` 填 `0` 到 `5` 的整数。
- `rationale` 要结合轨迹、代码改动、测试和最终答复写具体理由。
- 不要提交所有 `score = 0` 的空模板。

## 任务索引

| ID | 中文任务 | Qwen 文件 | Claude 文件 |
| --- | --- | --- | --- |
| CT-0001 | YongFu 游客端生产环境 fallback 收口 | `qwen/CT-0001.quality.toml` | `claude/CT-0001.quality.toml` |
| CT-0002 | YongFu 后台发布状态入口唯一化 | `qwen/CT-0002.quality.toml` | `claude/CT-0002.quality.toml` |
| CT-0003 | YongFu 新增 CMS 视频区块 | `qwen/CT-0003.quality.toml` | `claude/CT-0003.quality.toml` |
| CT-0004 | YongFu 3D 景点模型查看器内存释放 | `qwen/CT-0004.quality.toml` | `claude/CT-0004.quality.toml` |
| CT-0005 | AI PR Review 支持更多 PR URL 输入格式 | `qwen/CT-0005.quality.toml` | `claude/CT-0005.quality.toml` |
| CT-0006 | AI PR Review 大 PR 上下文降噪 | `qwen/CT-0006.quality.toml` | `claude/CT-0006.quality.toml` |
| CT-0007 | AI PR Review 风险项增加 evidence 展示 | `qwen/CT-0007.quality.toml` | `claude/CT-0007.quality.toml` |
| CT-0008 | 408 题库批量导入预检与错误提示 | `qwen/CT-0008.quality.toml` | `claude/CT-0008.quality.toml` |
| CT-0009 | 408 院校库组合筛选 | `qwen/CT-0009.quality.toml` | `claude/CT-0009.quality.toml` |
| CT-0010 | study-mobile 请求错误与登录失效处理 | `qwen/CT-0010.quality.toml` | `claude/CT-0010.quality.toml` |

## 7 个评分维度

| name | 中文含义 | 权重 | 主要看什么 |
| --- | --- | ---: | --- |
| `user_experience_and_interaction` | 用户体验与交互 | 1.0 | 页面状态、错误提示、交互是否清晰，用户是否能理解结果 |
| `task_planning_and_execution_control` | 任务规划与执行管控 | 1.0 | 是否系统性探索代码、是否按步骤推进、是否避免只改表面 |
| `semantic_understanding_and_logical_reasoning` | 语义理解与逻辑推理 | 1.0 | 是否真正理解业务风险、数据流、边界条件和因果关系 |
| `instruction_compliance_and_constraint_adherence` | 指令遵从与约束保持 | 1.0 | 是否满足题目中明确列出的要求和限制 |
| `engineering_quality_and_completeness` | 工程化质量与完备性 | 1.0 | 代码结构、类型、复用、测试覆盖、回归风险控制 |
| `delivery_completeness_and_usability` | 交付完整性与可用性 | 1.0 | 是否给出验证命令、测试结果、手动验证步骤，交付是否可复核 |
| `architecture_boundaries_and_security_compliance` | 架构边界与安全合规性 | 1.0 | 是否遵守模块边界，是否避免安全、权限、数据泄露或危险渲染问题 |

## 分数建议

- `5`：完整满足该维度，有清晰验证或证据。
- `4`：基本满足，只有小遗漏或小瑕疵。
- `3`：主流程部分满足，但有明显遗漏、边界缺失或验证不足。
- `2`：只做了表面修改，关键要求缺失。
- `1`：几乎没有有效完成该维度。
- `0`：未尝试、方向错误、无法运行，或引入严重问题。

## passrate 计算

```text
passrate = 7 个 score 的总和 / 35
```

也可以在 `D:\A3Code\Coding Trajectory` 下执行：

```powershell
python rubrics_templates\calc_passrate.py rubrics_templates\qwen rubrics_templates\claude
```

建议目标：

- Qwen：`passrate < 0.7`
- Claude：`passrate > 0.7`
- Claude：同一条任务下高于 Qwen。
