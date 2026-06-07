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
5. prompt_qwen 和 prompt_claude 描述同一个任务但措辞略有不同。followups_qwen: 2-3 条。followups_claude: 4-5 条
6. 不要以"你正在一个本地项目目录中工作"等模板化前缀开头，直接说需求
7. 渐进流程：初始 prompt = 大目标 → followup 1 = 核心实现 → followup 2 = 增强/边界 → followup 3+ = 测试、完善、文档

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

对于每个任务，在 `{{RUBRICS_DIR}}/qwen/` 和 `{{RUBRICS_DIR}}/claude/` 下各创建一个 `任务ID.quality.toml` 文件，包含 7 个评分标准：

```toml
[[criterion]]
name = "user_experience_and_interaction"
description = "Evaluates whether ...（根据任务内容编写）"
type = "likert"
points = 5
weight = 1.0
score = 0
rationale = ""

[[criterion]]
name = "task_planning_and_execution_control"
description = "Evaluates whether ..."
type = "likert"
points = 5
weight = 1.0
score = 0
rationale = ""

[[criterion]]
name = "semantic_understanding_and_logical_reasoning"
description = "Evaluates whether ..."
type = "likert"
points = 5
weight = 1.0
score = 0
rationale = ""

[[criterion]]
name = "instruction_compliance_and_constraint_adherence"
description = "Evaluates whether ..."
type = "likert"
points = 5
weight = 1.0
score = 0
rationale = ""

[[criterion]]
name = "engineering_quality_and_completeness"
description = "Evaluates whether ..."
type = "likert"
points = 5
weight = 1.0
score = 0
rationale = ""

[[criterion]]
name = "delivery_completeness_and_usability"
description = "Evaluates whether ..."
type = "likert"
points = 5
weight = 1.0
score = 0
rationale = ""

[[criterion]]
name = "architecture_boundaries_and_security_compliance"
description = "Evaluates whether ..."
type = "likert"
points = 5
weight = 1.0
score = 0
rationale = ""
```

每个 description 必须以 "Evaluates whether" 开头，针对当前任务的具体内容编写。qwen 和 claude 两个目录下同一任务的 rubric 内容相同。

## 任务 ID

按顺序使用以下 ID：{{TASK_IDS}}

## 开始执行

现在请分析项目并生成上述所有文件。直接写入文件，不要只输出内容。
