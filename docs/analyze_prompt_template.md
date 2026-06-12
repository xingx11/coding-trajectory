你是一个编码任务设计师。请分析当前项目目录的代码结构，然后为该项目设计 {{COUNT}} 个编码任务。

## 任务要求

每个任务需要按照下面的 slot 分配来设计（按顺序对应）：
{{SLOTS}}

## 分析步骤

1. 先浏览项目的 README、目录结构、依赖文件、核心源码，理解项目功能
2. 根据每个 slot 的 task_type 设计对应类型的任务（bug-fix = 修复 bug，feature = 新功能，enhancement = 增强，testing-quality = 测试质量，code-explanation = 代码解释分析）
3. 任务必须基于项目的真实代码，涉及跨文件理解，有明确的验收标准

## 提示词规则（极其重要）

prompt_qwen、prompt_claude、followups_qwen、followups_claude 必须满足：

1. 全部用中文（简体中文）
2. 像真人开发者在 AI 编码助手中打字一样 — 简短、口语化、自然
3. 初始 prompt（prompt_qwen / prompt_claude）只写一句话描述目标，如："帮我给这个项目加一个命令行状态查看功能"、"这个项目有个并发 bug，帮我修一下"
4. followups 是渐进式追问 — 每条只问一个具体的下一步，像自然对话："现在加上颜色显示"、"再写几个测试"、"处理一下边界情况"。每条 1-2 句话
5. prompt_qwen 和 prompt_claude 描述同一个任务但措辞略有不同。followups_qwen 和 followups_claude 数量必须相同（3-4 条）
6. 不要以"你正在一个本地项目目录中工作"等模板化前缀开头，直接说需求
7. 渐进流程：初始 prompt = 大目标 → followup 1 = 核心实现 → followup 2 = 增强/边界 → followup 3+ = 测试、完善、文档
8. prompt_qwen 和 prompt_claude 必须传达相同的信息量和难度。不要给某个模型额外的提示、文件路径或技术细节。两个模型应凭自身能力成功或失败，而非信息不对称
9. 两个模型的 followups 必须按相同顺序覆盖相同功能领域。措辞可以不同但实质要求必须等价
10. 不要在 prompt 中引用项目中实际不存在的文件、函数或技术细节

## 输出要求

对于每个任务，你需要直接写入两类文件：

### 文件 1：追加到 tasks.toml

将以下格式的条目追加到 `{{TASKS_TOML_PATH}}`（注意 Windows 路径中的反斜杠要双重转义）：

```toml
[[task]]
id = "任务ID"
project_path = "{{PROJECT_PATH_ESCAPED}}"
clone_method = "git"
task_type = "对应的task_type"
domain = "对应的domain"
language = "对应的language"
prompt_qwen = """中文一句话"""
followups_qwen = [
  "追问1",
  "追问2",
]
prompt_claude = """中文一句话"""
followups_claude = [
  "追问1",
  "追问2",
  "追问3",
  "追问4",
]
```

### 文件 2：创建 rubric 模板

对于每个任务，在 `{{RUBRICS_DIR}}/qwen/` 和 `{{RUBRICS_DIR}}/claude/` 下各创建一个 `任务ID.quality.toml` 文件。

**维度选择**：从以下 20 个候选维度中选择 7-10 个最能反映本任务质量差异的维度：

| name | 适用场景 |
|---|---|
| `user_experience_and_interaction` | 用户体验、交互节奏 |
| `task_planning_and_execution_control` | 计划、todo、执行控制 |
| `semantic_understanding_and_logical_reasoning` | 意图理解、代码定位、逻辑判断 |
| `instruction_compliance_and_constraint_adherence` | 用户约束、项目规则 |
| `engineering_quality_and_completeness` | 代码质量、测试意识 |
| `delivery_completeness_and_usability` | 最终产物、可运行性 |
| `architecture_boundaries_and_security_compliance` | 架构边界、安全合规（weight=2.0） |
| `tool_usage_and_failure_recovery` | 工具调用、失败恢复 |
| `evidence_grounding_and_trace_fidelity` | 证据一致性、虚假交付 |
| `testing_and_verification_rigor` | 测试、构建、lint |
| `context_exploration_and_code_navigation` | 文件读取、代码定位 |
| `requirements_clarification_and_scope_control` | 需求澄清、范围控制 |
| `environment_and_dependency_handling` | 依赖、环境、版本 |
| `attachment_and_artifact_handling` | PDF、图片、zip 等附件 |
| `external_research_and_source_use` | web search、GitHub、文档 |
| `custom_tool_and_protocol_compliance` | MCP、skill、内部工具 |
| `parallel_workflow_coordination` | 多模块、并行任务 |
| `security_privacy_and_secret_handling` | token、secret、auth |
| `maintainability_and_change_minimality` | diff 范围、可维护性 |
| `final_response_and_handoff_quality` | 最终回复、交接质量 |

选择规则：
1. 基于任务的具体特征选择（bug-fix 应考虑 `tool_usage_and_failure_recovery`、`testing_and_verification_rigor`；feature 应考虑 `context_exploration_and_code_navigation`、`maintainability_and_change_minimality` 等）
2. 如果某个维度对本任务没有可观察证据，不要选择它
3. `architecture_boundaries_and_security_compliance`（weight=2.0）通常应选择
4. 每个维度必须保持原子性：一个维度只评一件事，不能同时要求 A 和 B
5. 所选维度之间不能有实质性重叠。如果两个维度评价同一类能力，只保留更精确的一个

**description 定制化（极其重要）**：

每个选中维度的 description 必须根据当前项目和任务量身定制：

1. **保留 1-5 分档位结构**：必须包含 1 分到 5 分各档位的具体定义
2. **融入项目特征**：把项目名称、技术栈、具体问题融入各档位描述中
3. 不能使用通用模板，必须让人一看就知道这是针对什么项目什么任务的评分标准
4. 使用中文，写成一行 TOML 字符串
5. 每个 description 的各档位定义中，每档只描述一个判断条件。禁止在某一档中用"并且/且/同时"连接多个独立条件

定制化示例（以 turbulenz_engine 键盘事件修复任务为例）：

```toml
[[criterion]]
name = "semantic_understanding_and_logical_reasoning"
description = "在 turbulenz_engine 输入设备键盘事件重复触发修复任务中，模型对 onFocusIn/onFocusOut 事件注册链路的理解和修复逻辑推理是否准确？1分：完全误解键盘事件重复触发的根因，把问题归到无关模块。2分：定位到 inputapp 但修复方案不对，函数引用不一致导致 removeEventListener 无效。3分：理解主链路但遗漏鼠标/触摸事件的类似问题。4分：正确定位并修复键盘事件链路，函数引用一致。5分：精准定位 inputapp.ts 和 inputdevice.ts 的事件注册逻辑，用最小改动修复且覆盖所有输入类型"
type = "likert"
points = 5
weight = 1.0
score = 0
rationale = ""
```

不合格示例（通用模板，禁止使用）：

```toml
description = "模型理解意图并进行逻辑推理的准确性如何？1分：完全误解意图...5分：精准整合上下文..."
```

其他字段固定值：`type = "likert"`、`points = 5`、`score = 0`、`rationale = ""`。
`weight`：`architecture_boundaries_and_security_compliance` 为 `2.0`，其余为 `1.0`。

qwen 和 claude 两个目录下同一任务的 rubric 内容必须相同（相同维度、相同定制化 description）。

## 任务 ID

按顺序使用以下 ID：{{TASK_IDS}}

## 开始执行

现在请分析项目并生成上述所有文件。直接写入文件，不要只输出内容。
