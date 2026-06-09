# Qwen / Claude 人工 Rubrics 评分模板说明

这是 10 条 Coding Trajectory 投标样例的正式人工评分模板。

当前模板按 `docs/examples/` 中的样例文件口径整理：

- `qwen/` 下 Qwen 评分文件。
- `claude/` 下 Claude 评分文件。
- 每个文件 7-10 个评分维度（AI 在 `gen` 或 `score` 阶段从 20 个候选维度中选取）。
- 每个维度 `points = 5`。
- 大部分维度 `weight = 1.0`（`architecture_boundaries_and_security_compliance` 为 2.0）。
- `passrate = sum(score × weight) / sum(points × weight)`。

## 字段从哪里来

- `name`：评分维度的英文标识，AI 根据任务特征从 20 个候选维度中选取。
- `description`：AI 自动生成的定制化评分标准，融入项目名称、技术栈和具体任务特征，包含 1-5 分各档位的具体定义。不接受通用模板化的 description。
- `type`：固定为 `likert`，表示按 1-5 分打分。
- `points`：固定为 `5`，与官方示例一致。
- `weight`：大部分为 `1.0`（`architecture_boundaries_and_security_compliance` 为 2.0）。
- `score`：AI 评分或人工审核后填写。
- `rationale`：评分理由，需写具体证据。

`name` 和 `description` 在 `gen` 阶段由 AI 自动生成（针对每个任务的项目和需求定制化）。`score` 和 `rescore` 命令可基于轨迹数据进一步优化。

## 填写规则

- 只填写 `score` 和 `rationale`。
- 不建议修改 `name`、`description`、`type`、`points`、`weight`。
- 每个 `score` 填 `1` 到 `5` 的整数。
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

## 评分维度

`gen` 命令在生成任务时会自动从 20 个候选维度中选择 7-10 个最适合当前任务的维度，并生成定制化的 description。同一任务的 Qwen 和 Claude 使用相同的维度集合。`score` 和 `rescore` 命令可基于轨迹数据进一步优化维度选择和 description。

| name | 中文含义 | 权重 | 主要看什么 |
| --- | --- | ---: | --- |
| `user_experience_and_interaction` | 用户体验与交互 | 1.0 | 页面状态、错误提示、交互是否清晰，用户是否能理解结果 |
| `task_planning_and_execution_control` | 任务规划与执行管控 | 1.0 | 是否系统性探索代码、是否按步骤推进、是否避免只改表面 |
| `semantic_understanding_and_logical_reasoning` | 语义理解与逻辑推理 | 1.0 | 是否真正理解业务风险、数据流、边界条件和因果关系 |
| `instruction_compliance_and_constraint_adherence` | 指令遵从与约束保持 | 1.0 | 是否满足题目中明确列出的要求和限制 |
| `engineering_quality_and_completeness` | 工程化质量与完备性 | 1.0 | 代码结构、类型、复用、测试覆盖、回归风险控制 |
| `delivery_completeness_and_usability` | 交付完整性与可用性 | 1.0 | 是否给出验证命令、测试结果、手动验证步骤，交付是否可复核 |
| `architecture_boundaries_and_security_compliance` | 架构边界与安全合规性 | **2.0** | 是否遵守模块边界，是否避免安全、权限、数据泄露或危险渲染问题 |

## 分数建议

- `5`：完整满足该维度，有清晰验证或证据。
- `4`：基本满足，只有小遗漏或小瑕疵。
- `3`：主流程部分满足，但有明显遗漏、边界缺失或验证不足。
- `2`：只做了表面修改，关键要求缺失。
- `1`：几乎没有有效完成该维度。
- `0`：未填写（空模板状态）。

## passrate 计算

```text
passrate = sum(score × weight) / sum(points × weight)
```

也可以在 `D:\A3Code\Coding Trajectory` 下执行：

```powershell
python rubrics_templates\calc_passrate.py rubrics_templates\qwen rubrics_templates\claude
```

建议目标：

- Qwen：`passrate < 0.7`
- Claude：`passrate > 0.7`
- Claude：同一条任务下高于 Qwen。
