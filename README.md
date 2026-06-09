# Coding Trajectory 基准数据生产工作区

本仓库是一个数据生产工作区，用于生成 Coding Trajectory 基准测试样本，比较 Qwen 和 Claude 在真实本地编码任务中的表现。

每条样本包含：一对 Qwen/Claude 的 trajectory JSONL、一对评分文件、任务元数据，以及写入 submission.csv 的汇总行。

---

## 目录结构

```text
.
├─ ctpipe/                    # 自动化 pipeline 源码
│  ├─ __init__.py             # 包初始化
│  ├─ __main__.py             # `python -m ctpipe` 入口
│  ├─ cli.py                  # 命令行入口
│  ├─ config.py               # tasks.toml + .env 配置加载
│  ├─ distribution.py         # 225 行题目分布表 + 加权采样
│  ├─ github_search.py        # GitHub REST API 项目搜索 + clone
│  ├─ gen.py                  # AI 全自动任务生成（含 clone-only / analyze 模式）
│  ├─ prepare.py              # 项目克隆 + 交付目录骨架
│  ├─ run.py                  # claude -p 多轮执行
│  ├─ collect.py              # trajectory JSONL 收集
│  ├─ score.py                # AI 自动评分（含重试）
│  ├─ finalize.py             # passrate 计算 + submission.csv
│  ├─ validate.py             # 完整性校验
│  ├─ rescore.py              # 定制化维度重评分（两轮 AI 调用）
│  ├─ check.py                # 深度验证（轮数、模型、session、评分内容）
│  ├─ clean.py                # 交付后清理（runs、缓存、旧批次）
│  ├─ export.py               # 导出交付结果为结构化 JSON 报告
│  ├─ retry.py                # 自动重试失败/部分完成的任务
│  ├─ stats.py                # Pipeline 各阶段统计信息
│  ├─ state.py                # JSON 状态管理（幂等重跑，线程安全）
│  ├─ trajectory.py           # JSONL 解析工具
│  ├─ toml_utils.py           # TOML 评分文件读写
│  ├─ project_scan.py         # 项目目录扫描（README、依赖、目录树）
│  └─ project_hash.py         # Windows 路径 → Claude 项目 hash
├─ docs/                      # 参考文档、模板、示例
│  ├─ examples/               #   参考 JSONL 和 TOML 示例
│  ├─ analyze_prompt_template.md  #   --analyze 模式的提示词模板
│  ├─ metadata_template.md    #   metadata 文件模板
│  ├─ submission_template.csv #   submission.csv 表头模板
│  ├─ 评分规范.md              #   评分规则和标准详细说明
│  ├─ 评分文件规范.md          #   评分 TOML 文件格式规范
│  ├─ collect-salvage-force.md #   collect 阶段的抢救和强制收集指南
│  ├─ 对外-出题指南 Coding Rubrics.pdf   # 需求文档
│  └─ 对外-数据需求文档 Coding Rubrics.pdf # 需求文档
├─ rubrics_templates/         # 评分模板目录 + passrate 工具
│  ├─ calc_passrate.py        #   passrate 计算脚本
│  ├─ qwen/                   #   Qwen 评分模板（由 gen 自动生成）
│  └─ claude/                 #   Claude 评分模板（由 gen 自动生成）
├─ delivery_YYYYMMDD/         # 交付批次（由 pipeline 自动创建）
├─ tasks.toml                 # 任务配置（核心，由 gen 生成或手动编写）
├─ .env                       # API 密钥（不提交，需从 .env.template 创建）
└─ .env.template              # 环境变量模板
```

---

## 环境准备

### 前置条件

- **Python 3.11+**（需要 `tomllib`）
- **Claude Code CLI** 已安装并可在终端执行 `claude -p`
- **Git** 已安装

### 网络代理配置（中国大陆必需）

`gen` 命令需要访问 GitHub API 和 `git clone`。如果你在中国大陆，需要配置 Git 代理：

```powershell
# 设置 HTTP/HTTPS 代理（端口号改为你的代理端口）
git config --global http.proxy http://127.0.0.1:7897
git config --global https.proxy http://127.0.0.1:7897

# 切换 SSL 后端为 Windows 原生（解决 OpenSSL 与代理的 TLS 兼容性问题）
git config --global http.sslBackend schannel
```

> **注意**：代理地址必须用 `http://` 而不是 `https://`（连接本地代理不需要 TLS）。如果不设置 `sslBackend schannel`，即使代理正确也会出现 `TLS connect error: unexpected eof while reading` 错误。

> **重要**：以上 Git 代理配置仅对 `git clone` 生效。`gen` 命令的 GitHub API 搜索使用 Python `urllib`，需要在 `.env` 中单独配置 `HTTP_PROXY`（见下方环境变量配置）。两者缺一不可。

### 配置 tasks.toml 路径

`tasks.toml` 中有两个关键路径需要根据你的环境修改：

```toml
[batch]
delivery_date = "20260605"        # 交付批次日期，自动生成 delivery_YYYYMMDD/ 目录
runs_root = "D:\\A3Code\\runs"     # 项目克隆和运行的根目录（改为你的路径）
max_parallel = 6                  # 并发任务数
```

**`runs_root` 说明**：Pipeline 运行时会在此目录下为每个任务创建隔离的项目副本：

```text
runs_root/
├─ _projects/                  # gen --per-project 批量模式的共享源码
│  └─ bareiron/                #   各任务的 project_path 指向这里
├─ CT-0001/                    # gen 单任务模式克隆的源码
│  └─ Yongfu-Web/
├─ CT-0001-qwen/               # CT-0001 任务的 Qwen 运行目录（prepare 从源项目克隆）
├─ CT-0001-claude/             # CT-0001 任务的 Claude 运行目录（prepare 从源项目克隆）
├─ CT-0002-qwen/
├─ CT-0002-claude/
└─ ...
```

使用 `gen` 命令时，GitHub 项目会克隆到 `runs_root` 下：单任务模式放在 `CT-xxxx/` 子目录，批量模式放在 `_projects/` 子目录。可通过 `--clone-dir` 指定其他目录。

### 配置环境变量

```powershell
copy .env.template .env
```

编辑 `.env`，填入真实的 API 密钥：

```env
QWEN_AUTH_TOKEN=<your-qwen-token>
QWEN_BASE_URL=<your-qwen-base-url>
QWEN_MODEL=qwen3.7-max

CLAUDE_AUTH_TOKEN=<your-claude-token>
CLAUDE_BASE_URL=<your-claude-base-url>
CLAUDE_MODEL=claude-opus-4-6-20260205

# 可选但推荐：GitHub Token（提高 gen 命令的 API 限额）
# 无 Token：10 次搜索/分钟；有 Token：30 次搜索/分钟
# 创建地址：https://github.com/settings/tokens（无需勾选任何权限，public repo 搜索不需要 scope）
GITHUB_TOKEN=<your-github-token>

# 可选：Gitee Token（gen --source gitee 必需，搜索 API 要求认证）
# 创建地址：https://gitee.com/personal_access_tokens
GITEE_TOKEN=<your-gitee-token>

# HTTP 代理（中国大陆必需，用于 GitHub API 搜索和 git clone）
# 注意：Gitee 不需要代理
# 示例：http://127.0.0.1:7897
HTTP_PROXY=<your-proxy-url>
```

### 技术原理：Qwen 如何通过 Claude Code 运行

本项目使用 Claude Code CLI（`claude -p`）作为统一的执行引擎，Qwen 和 Claude 都通过它运行。切换模型的方式是设置环境变量：

| 环境变量 | 作用 | 示例值 |
|----------|------|--------|
| `ANTHROPIC_AUTH_TOKEN` | API 认证令牌 | `.env` 中的 `QWEN_AUTH_TOKEN` |
| `ANTHROPIC_BASE_URL` | API 端点地址 | `.env` 中的 `QWEN_BASE_URL` |
| `ANTHROPIC_MODEL` | 主模型名称 | `qwen3.7-max` |
| `ANTHROPIC_DEFAULT_HAIKU_MODEL` | Haiku 模型覆盖 | 同上，防止 fallback 到 Claude |
| `ANTHROPIC_DEFAULT_SONNET_MODEL` | Sonnet 模型覆盖 | 同上 |
| `ANTHROPIC_DEFAULT_OPUS_MODEL` | Opus 模型覆盖 | 同上 |
| `CLAUDE_CODE_SUBAGENT_MODEL` | 子代理模型覆盖 | 同上 |
| `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC` | 禁用遥测等非必要请求 | `1` |

> **Pipeline 自动处理**：`python -m ctpipe run` 会根据 `.env` 中的 `QWEN_*` / `CLAUDE_*` 配置自动设置以上环境变量，无需手动操作。

> **手动运行 Qwen**：如果想在终端直接用 Claude Code 运行 Qwen，需要设置以上环境变量：

```powershell
# PowerShell 示例（端口和模型名改为你的实际值）
$env:ANTHROPIC_AUTH_TOKEN = "<your-qwen-token>"
$env:ANTHROPIC_BASE_URL = "<your-qwen-base-url>"
$env:ANTHROPIC_MODEL = "qwen3.7-max"
$env:ANTHROPIC_DEFAULT_HAIKU_MODEL = "qwen3.7-max"
$env:ANTHROPIC_DEFAULT_SONNET_MODEL = "qwen3.7-max"
$env:ANTHROPIC_DEFAULT_OPUS_MODEL = "qwen3.7-max"
$env:CLAUDE_CODE_SUBAGENT_MODEL = "qwen3.7-max"
$env:CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC = "1"

claude --dangerously-skip-permissions --setting-sources local --model "$env:ANTHROPIC_MODEL"
```

> **注意**：必须同时覆盖所有模型变量（Haiku/Sonnet/Opus/Subagent），否则 Claude Code 在某些内部调用中会 fallback 到默认的 Claude 模型，导致请求发送到错误的端点。Pipeline 会自动将 API 端点主机名加入 `NO_PROXY`，确保 API 请求不经过代理。

---

## 完整开发流程

从零开始到产出一个可交付批次，完整流程分为三大阶段：**任务准备（可全自动） → 自动化执行 → 人工审核**。

### 第一阶段：任务准备

任务准备支持两种方式：**全自动生成**（推荐）和**人工编写**。

#### 方式一：全自动生成（推荐）

使用 `gen` 子命令，基于需求文档中的 225 行题目分布表自动完成全部准备工作：

```powershell
# 按分布自动生成 45 条任务（每个项目生成 5 条，共克隆 9 个仓库）
python -m ctpipe gen --count 45 --per-project 5

# 单项目模式（每个项目 1 条任务，共克隆 45 个仓库）
python -m ctpipe gen --count 45

# 指定领域和语言
python -m ctpipe gen --count 10 --domain web_frontend --language ts

# 指定任务类型
python -m ctpipe gen --count 5 --task-type bug-fix

# 从已有本地项目生成（不搜索 GitHub）
python -m ctpipe gen --count 3 --from-local "D:\A3Code\YongFu\Yongfu-Web"

# 从 Gitee 搜索项目（国内直连，不需要代理）
python -m ctpipe gen --count 45 --per-project 3 --source gitee

# 仅克隆仓库（不调用 AI，后续手动分析）
python -m ctpipe gen --clone-only --count 6 --per-project 3

# 用 Claude Code 直接分析已克隆仓库，生成任务条目和评分模板
python -m ctpipe gen --from-local "D:\A3Code\runs\_projects\repo_name" --count 3 --analyze

# 预览模式（不 clone、不写文件、不调用 AI）
python -m ctpipe gen --count 45 --per-project 3 --dry-run
```

`gen` 的完整流程：

1. **加权采样**：从 225 行分布表中按权重随机选取 N 个 `(task_type, domain, language)` 组合
2. **分组**（`--per-project > 1` 时）：将 N 个 slot 按每组 M 个分组，每组共享一个项目
3. **搜索项目**：根据 domain 和 language，通过 GitHub/Gitee REST API 搜索合适的开源项目（GitHub: stars >= 50、Gitee: stars >= 30、非 fork、非 archived、< 100MB）
4. **克隆项目**：`git clone --depth 1 --filter=blob:none` 到指定目录
5. **扫描项目**：提取 README、目录树、依赖文件，生成 ~1.5K 字符的项目概要
6. **AI 发现任务**：
   - 单任务模式（`per_project=1`）：两阶段 — idea 生成 + expand（共 2 次 API 调用）
   - 批量模式（`per_project>1`）：先批量生成 M 个 idea（1 次调用），再逐个 expand（M 次调用），共 M+1 次
7. **写入配置**：
   - 自动递增 task_id（`CT-xxxx`）
   - 持久化 AI 生成的 `task_title`、`task_description`、`acceptance_criteria` 到 `tasks.toml`（供下游评分使用）
   - AI 从 20 个候选维度中选择 7-10 个，为每个维度生成融入项目特征的定制化 description（含 1-5 分档位定义），写入 `rubrics_templates/qwen/` 和 `rubrics_templates/claude/`
   - 格式化 `[[task]]` 条目追加到 `tasks.toml`

**分步模式**：当 AI 调用超时频繁时，可拆分为两步：

1. **仅克隆**：`python -m ctpipe gen --clone-only --count 6 --per-project 3` — 只搜索+克隆仓库，输出路径列表到 `_cloned_repos.json`
2. **分析生成**（二选一）：
   - **自动化**：`python -m ctpipe gen --from-local "<path>" --count 3 --analyze` — 让 Claude Code 在仓库目录下分析并直接写入 tasks.toml 和 rubric 文件
   - **交互式**：在 Claude Code 中使用 `docs/analyze_prompt_template.md` 作为提示词（详见下方说明）

**交互式分析操作指南**：

当你克隆了仓库后，想通过 Claude Code 手动分析项目并将任务写入 `tasks.toml` 时：

1. **在目标仓库目录下打开 Claude Code**：
   ```powershell
   cd D:\A3Code\runs\_projects\repo_name
   claude
   ```

2. **发送分析提示词**，将 `docs/analyze_prompt_template.md` 的内容作为 prompt，替换其中的变量：

   | 变量 | 含义 | 示例值 |
   |------|------|--------|
   | `{{COUNT}}` | 要生成的任务数 | `3` |
   | `{{SLOTS}}` | 任务类型分配列表 | `1. task_type=bug-fix, domain=web_frontend, language=ts` |
   | `{{TASKS_TOML_PATH}}` | tasks.toml 的绝对路径 | `D:\\A3Code\\Coding Trajectory\\tasks.toml` |
   | `{{PROJECT_PATH_ESCAPED}}` | 项目路径（双反斜杠） | `D:\\A3Code\\runs\\_projects\\repo_name` |
   | `{{RUBRICS_DIR}}` | rubric 模板目录路径 | `D:\\A3Code\\Coding Trajectory\\rubrics_templates` |
   | `{{TASK_IDS}}` | 任务 ID 列表 | `CT-0010, CT-0011, CT-0012` |

3. **快捷方式**：也可以直接在 Claude Code 中引用文件：
   ```
   请按照 @docs/analyze_prompt_template.md 的模板分析当前项目，生成 3 个任务。
   SLOTS: 1. bug-fix/web_frontend/ts  2. feature/web_frontend/ts  3. enhancement/web_frontend/ts
   TASKS_TOML_PATH: D:\A3Code\Coding Trajectory\tasks.toml
   PROJECT_PATH: D:\A3Code\runs\_projects\repo_name
   RUBRICS_DIR: D:\A3Code\Coding Trajectory\rubrics_templates
   TASK_IDS: CT-0010, CT-0011, CT-0012
   ```

   Claude Code 会读取模板文件，分析项目代码结构，然后自动将任务条目追加到 `tasks.toml` 并在 `rubrics_templates/` 下创建对应的评分模板文件。

**性能对比**（以 45 条任务为例）：

| 模式 | 克隆仓库数 | API 调用数 | 预估耗时 |
|:---:|:---:|:---:|:---:|
| `--per-project 1`（默认） | 45 | 90 | ~225 分钟 |
| `--per-project 3` | 15 | 60 | ~75 分钟 |

防重复机制：已使用的 GitHub 仓库记录在 `pipeline_state.json` 中，搜索时自动排除。

#### 方式二：人工编写

如果需要精确控制任务内容，也可以手动完成以下步骤：

1. **选择源项目**：从本地代码库中选择真实项目作为任务载体
2. **设计任务**：确定任务 ID（`CT-xxxx`）、任务类型、Prompt 和 Follow-ups
3. **编写评分模板**：每个任务 2 份（Qwen/Claude 各一份），放在 `rubrics_templates/` 下
4. **写入 tasks.toml**：将任务配置追加到 `tasks.toml`

评分模板默认包含 7-10 个维度（从 20 个候选维度中选取），每个维度 1-5 分。`gen` 命令在生成任务时会自动调用 AI 生成定制化的评分维度和 description，包含融入项目名称、技术栈和具体问题的 1-5 分档位定义。AI 自动评分时同一任务的 Qwen 和 Claude 使用相同的维度集合（成对评分）。

**定制化评分**：每个维度的 description 必须根据当前项目和任务量身定制。`gen` 命令（包括普通模式和 `--analyze` 模式）、`score` 命令和 `rescore` 命令都会自动生成定制化 description。通用模板化的 description（如"模型理解意图的准确性如何"）不合格，正确示例：

> "在 turbulenz_engine 输入设备键盘事件重复触发修复任务中，模型对事件注册链路的理解和修复逻辑推理是否准确？1分：完全误解根因...2分：定位到模块但修复方案不对...3分：理解主链路但遗漏类似问题...4分：正确修复主链路...5分：精准定位并用最小改动覆盖所有输入类型"

20 个候选维度涵盖：

| 类别 | 维度 |
|------|------|
| 交互与理解 | user_experience_and_interaction, semantic_understanding_and_logical_reasoning, requirements_clarification_and_scope_control |
| 规划与执行 | task_planning_and_execution_control, instruction_compliance_and_constraint_adherence |
| 工程质量 | engineering_quality_and_completeness, delivery_completeness_and_usability, maintainability_and_change_minimality |
| 工具与验证 | tool_usage_and_failure_recovery, testing_and_verification_rigor, context_exploration_and_code_navigation |
| 安全与架构 | architecture_boundaries_and_security_compliance (weight=2.0), security_privacy_and_secret_handling |
| 特殊场景 | environment_and_dependency_handling, attachment_and_artifact_handling, external_research_and_source_use, custom_tool_and_protocol_compliance, parallel_workflow_coordination |
| 证据与交接 | evidence_grounding_and_trace_fidelity, final_response_and_handoff_quality |

`tasks.toml` 条目格式：

```toml
[batch]
delivery_date = "20260603"
runs_root = "D:\\A3Code\\runs"
max_parallel = 3

[[task]]
id = "CT-0001"
project_path = "D:\\A3Code\\YongFu\\Yongfu-Web"
clone_method = "git"
task_type = "bug-fix"
domain = "web_frontend"
language = "ts"
task_title = "修复导航栏在移动端横屏时的布局溢出问题"
task_description = "移动端横屏模式下导航栏内容超出视口宽度，导致水平滚动条出现"
acceptance_criteria = [
  "横屏模式下导航栏不超出视口",
  "竖屏模式不受影响",
]
bad_pattern = "lazy_shortcut"
prompt_qwen = """..."""
prompt_claude = """..."""
followups_qwen = ["...", "..."]
followups_claude = ["...", "...", "..."]
```

---

### 第二阶段：自动化执行（Pipeline）

可以一键运行全流程，也可以分步执行。

#### 一键执行

```powershell
python -m ctpipe --config tasks.toml --env .env all
```

等价于依次执行以下 6 个阶段：

#### Stage 1：Prepare — 项目克隆 + 交付骨架

```powershell
python -m ctpipe prepare
```

做了什么：
1. 创建 `delivery_YYYYMMDD/` 交付目录骨架（trajectories、scores、metadata 子目录）
2. 为每个任务从 `project_path` 克隆 2 份独立源项目到隔离的运行目录（`runs_root/CT-xxxx-qwen/`、`runs_root/CT-xxxx-claude/`），保证两侧基线完全一致
3. 在每个运行目录写入 `.claude/settings.local.json`，授予 Claude Code 全量工具权限
4. 将 rubric 模板复制到交付目录 scores 下
5. 初始化 `submission.csv` 模板
6. 已完成的任务在重跑时会验证运行目录是否仍存在，如被清理则自动重新克隆

#### Stage 2：Run — 执行编码任务

```powershell
python -m ctpipe run
```

做了什么：
1. 对每个任务，同时启动 Qwen 和 Claude 两个 `claude -p` 进程
2. 第 1 轮发送初始 prompt，后续轮通过 `--resume <session_id>` 发送 follow-up
3. 通过 `asyncio.Semaphore(max_parallel)` 控制最大并发任务数
4. 每轮有 `turn_timeout`（默认 900s），整体有 `total_timeout`（默认 3600s）
5. 运行结果（session_id、耗时、轮数）记录到 `pipeline_state.json`

核心机制：
- 两个模型通过环境变量切换（`ANTHROPIC_AUTH_TOKEN`、`ANTHROPIC_BASE_URL`、`ANTHROPIC_MODEL`）
- Qwen 和 Claude 共用 `claude -p` CLI，但指向不同的 API endpoint
- 失败的轮次不中断整个任务，session 会继续推进

#### Stage 3：Collect — 收集 Trajectory

```powershell
python -m ctpipe collect
```

做了什么：
1. 根据运行目录路径计算 Claude 项目 hash，定位 `~/.claude/projects/<hash>/` 下的 JSONL 文件
2. 按运行开始时间和 session_id 筛选匹配的 JSONL
3. 解析 JSONL 验证模型提供商（检测 model 字段中是否包含 "qwen" 或 "claude"）
4. 复制并重命名为标准格式 `CT-xxxx.jsonl`，放入交付目录

#### Stage 4：Score — AI 自动评分

```powershell
python -m ctpipe score
```

做了什么：
1. 从 trajectory JSONL 提取精简文本（用户消息 + 助手回复 + 工具调用摘要，上限 50K 字符）
2. 构建中文评分 prompt，包含任务背景（项目名、技术栈、任务标题、验收标准、项目概要）、评分维度定义、评分规则（1-5 分）、rationale 写作要求和 Bad Pattern 识别指引
3. 成对评分：先评 Qwen（AI 自选 7-10 个维度），再评 Claude（使用 Qwen 选定的相同维度），保证 passrate 可比
4. 通过 `claude -p` 调用 AI 评分，解析返回的 TOML 并写入交付目录
5. 验证输出：维度名合法性、score 1-5 整数、rationale 非空、维度数量 6-10
6. 解析失败时保存为 `.draft.txt` 供人工处理

#### Stage 5：Finalize — 汇总提交

```powershell
python -m ctpipe finalize
```

做了什么：
1. 读取所有评分文件，计算 passrate（`sum(score × weight) / sum(points × weight)`）
2. 检查阈值：qwen < 0.7、claude > 0.7、claude > qwen、(claude-qwen)/qwen > 25%
3. 从 trajectory 中提取 session_id
4. 生成 `submission.csv`，包含所有任务的 trajectory 路径、session_id、评分路径、passrate、任务标签

#### Stage 6：Validate — 完整性校验

```powershell
python -m ctpipe validate
```

校验内容：
- trajectory JSONL 文件存在且模型匹配
- score TOML 文件存在
- submission.csv 中的路径、session_id、passrate 与实际文件一致
- metadata 文件存在
- passrate 阈值：qwen < 0.7、claude > 0.7、claude > qwen、相对增益 > 25%

---

### 第三阶段：人工审核（人工）

Pipeline 产出的是 AI 初评结果，需要人工复核：

1. **审核评分**：打开 `delivery_YYYYMMDD/scores/` 下的 `.quality.toml` 文件，检查 AI 给出的 score 和 rationale 是否合理，手动修正
2. **检查 passrate 阈值**：确认每条数据满足 `qwen < 0.7`、`claude > 0.7`、`claude > qwen`、`(claude-qwen)/qwen > 25%`
3. **补充 metadata**：在 `metadata/CT-xxxx.md` 中记录任务背景和特殊说明
4. **重新 finalize + validate**：修正评分后重新运行以更新 submission.csv

```powershell
python -m ctpipe finalize
python -m ctpipe validate
```

#### 深度验证（check）

`validate` 只检查文件是否存在，`check` 进一步检查内容质量：

```powershell
python -m ctpipe check
python -m ctpipe check --tasks CT-0001 CT-0002
python -m ctpipe check --models qwen
```

检查内容：
- trajectory 轮数是否在合理范围内（2-8 轮）
- trajectory 中的模型标识是否与预期一致（qwen/claude）
- session_id 是否与 submission.csv 中记录的一致
- 评分文件是否完整（7-10 个维度，维度名合法，1-5 分范围，rationale 非空）
- passrate 是否与评分文件实际计算值一致
- 同一任务 qwen/claude 的维度集合是否一致（成对一致性检查）
- 跨模型阈值是否满足

#### 定制化重评分（rescore）

当评分文件使用通用模板 description 或 passrate 不达标时，使用 `rescore` 重新生成定制化维度和评分：

```powershell
# 重评分所有任务
python -m ctpipe rescore

# 重评分指定任务
python -m ctpipe rescore --tasks CT-0001 CT-0002

# 仅重评分某一侧
python -m ctpipe rescore --tasks CT-0001 --models qwen
```

`rescore` 的两轮流程：

1. **第一轮：维度选择 + description 定制** — 读取任务元数据（项目名、技术栈、任务标题、验收标准、项目概要）和两边轨迹摘要，AI 从 20 个候选维度中选 7-10 个，为每个维度写定制化 description（包含 1-5 分档位定义，融入项目特征）
2. **第二轮：评分** — 先评 Qwen（带 passrate < 0.7 约束），再评 Claude（带 passrate > 0.7 且 gap > 25% 约束），使用第一轮生成的定制维度，并注入任务背景（项目信息、验收标准等）辅助评分

结果写入 `delivery_*/scores/` 和 `rubrics_templates/`，并验证 passrate 阈值。

#### 自动重试（retry）

Pipeline 的任何阶段都可能因超时、限流等原因产生 `failed` 或 `partial` 状态。`retry` 自动检测这些异常并重试：

```powershell
# 自动重试所有失败/部分完成的任务
python -m ctpipe retry

# 仅重试指定阶段
python -m ctpipe retry --stages run collect

# 指定最大重试次数
python -m ctpipe retry --max-retries 3

# 预览模式（只打印，不执行）
python -m ctpipe retry --dry-run

# 不级联到下游阶段
python -m ctpipe retry --no-cascade
```

默认行为：检测失败条目 → 重置状态 → 重新执行 → 级联到受影响的下游阶段。例如 `run` 重试成功后会自动重跑 `collect` → `score` → `finalize`。

#### 统计信息（stats）

快速查看各阶段的完成/失败/待处理数量：

```powershell
# 表格形式（默认）
python -m ctpipe stats

# JSON 格式
python -m ctpipe stats --format json

# 包含 run/score 耗时统计
python -m ctpipe stats --timing
```

#### 导出报告（export）

将交付结果导出为结构化 JSON 文件，包含 batch_info、per-task 详情和 summary 汇总：

```powershell
# 导出到文件
python -m ctpipe export --output delivery_20260609/report.json

# 输出到 stdout
python -m ctpipe export

# 仅导出指定任务
python -m ctpipe export --tasks CT-0001 CT-0002
```

#### 交付后清理（clean）

数据交付上传后，清理中间文件释放磁盘空间：

```powershell
# 仅清理 runs/ 下的克隆目录（默认行为）
python -m ctpipe clean

# 同时清理 ~/.claude/projects/ 下的 JSONL trajectory 缓存
python -m ctpipe clean --cache

# 同时清理旧批次交付目录（保留当前批次）
python -m ctpipe clean --old-deliveries

# 全部清理
python -m ctpipe clean --cache --old-deliveries

# 预览模式（只打印，不删除）
python -m ctpipe clean --dry-run

# 跳过 runs/ 清理（仅清理缓存或旧批次）
python -m ctpipe clean --no-runs --cache
```

---

## 常用命令参考

| 命令 | 说明 |
|------|------|
| `python -m ctpipe gen --count 45 --per-project 3` | 全自动生成 45 条任务（15 个仓库，每个 3 条） |
| `python -m ctpipe gen --count 45` | 全自动生成 45 条任务（45 个仓库，每个 1 条） |
| `python -m ctpipe gen --count 45 --per-project 3 --source gitee` | 从 Gitee 搜索项目（国内免代理） |
| `python -m ctpipe gen --clone-only --count 6 --per-project 3` | 仅搜索+克隆仓库（跳过 AI 分析） |
| `python -m ctpipe gen --from-local "<path>" --count 3 --analyze` | Claude Code 直接分析仓库并写入任务 |
| `python -m ctpipe all` | 运行全流程（prepare→run→collect→score→finalize→validate） |
| `python -m ctpipe prepare` | 仅克隆项目 + 创建交付骨架 |
| `python -m ctpipe run --tasks CT-0001 CT-0002` | 运行指定任务 |
| `python -m ctpipe run --models qwen` | 仅运行 Qwen |
| `python -m ctpipe collect` | 收集 trajectory JSONL |
| `python -m ctpipe score` | AI 自动评分 |
| `python -m ctpipe finalize` | 计算 passrate + 生成 submission.csv |
| `python -m ctpipe validate` | 校验交付完整性 |
| `python -m ctpipe validate --models qwen` | 仅校验 Qwen 侧数据 |
| `python -m ctpipe check` | 深度验证（轮数、模型、session、评分内容） |
| `python -m ctpipe check --tasks CT-0001` | 深度验证指定任务 |
| `python -m ctpipe rescore` | 定制化维度重评分（两轮 AI：维度选择+评分） |
| `python -m ctpipe rescore --tasks CT-0001 CT-0002` | 重评分指定任务 |
| `python -m ctpipe clean` | 清理 runs/ 克隆目录（交付后释放磁盘） |
| `python -m ctpipe clean --cache` | 同时清理 ~/.claude/projects/ 缓存 |
| `python -m ctpipe clean --old-deliveries` | 同时清理旧批次交付目录 |
| `python -m ctpipe clean --dry-run` | 预览要删除的内容（不实际删除） |
| `python -m ctpipe reset --tasks CT-0001 --stages run collect` | 重置指定任务的指定阶段状态 |
| `python -m ctpipe stats` | 显示各阶段统计信息（表格/JSON） |
| `python -m ctpipe stats --timing` | 包含 run/score 耗时统计 |
| `python -m ctpipe retry` | 自动重试失败/部分完成的任务 |
| `python -m ctpipe retry --dry-run` | 预览要重试的内容（不实际执行） |
| `python -m ctpipe retry --max-retries 3` | 最多重试 3 次（默认 2） |
| `python -m ctpipe export --output report.json` | 导出交付结果为结构化 JSON 报告 |
| `python -m ctpipe export --tasks CT-0001` | 仅导出指定任务 |
| `python rubrics_templates\calc_passrate.py <path>` | 手动计算 passrate |

### 可选参数

```powershell
# 指定配置文件
python -m ctpipe --config tasks.toml --env .env all

# 指定任务子集和模型
python -m ctpipe run --tasks CT-0001 CT-0003 --models claude

# 调整超时
python -m ctpipe run --turn-timeout 900 --total-timeout 3600

# gen：按领域/语言/类型筛选
python -m ctpipe gen --count 10 --domain web_frontend --language ts --task-type bug-fix

# gen：每个项目生成多条任务（节省克隆和 API 开销）
python -m ctpipe gen --count 12 --per-project 4

# gen：从本地项目生成
python -m ctpipe gen --count 3 --from-local "D:\A3Code\YongFu\Yongfu-Web"

# gen：指定 clone 目录
python -m ctpipe gen --count 5 --clone-dir "D:\A3Code\cloned_projects"

# gen：调整单任务生成超时（默认 900s）
python -m ctpipe gen --count 5 --gen-timeout 1200

# gen：从 Gitee 搜索项目（国内直连，无需代理）
python -m ctpipe gen --count 45 --per-project 3 --source gitee

# gen：仅克隆仓库，不调 AI（拆分网络和分析步骤）
python -m ctpipe gen --clone-only --count 6 --per-project 3

# gen：用 Claude Code 分析本地仓库并直接写入任务文件
python -m ctpipe gen --from-local "D:\A3Code\runs\_projects\repo" --count 3 --analyze

# gen：预览模式（不 clone、不写文件、不调用 AI）
python -m ctpipe gen --count 45 --per-project 3 --dry-run

# check：深度验证指定任务
python -m ctpipe check --tasks CT-0001 CT-0002 --models qwen

# stats：显示统计信息（JSON 格式 + 耗时）
python -m ctpipe stats --format json --timing

# retry：自动重试失败任务（仅指定阶段，级联下游）
python -m ctpipe retry --stages run collect --max-retries 3

# retry：预览模式
python -m ctpipe retry --dry-run

# export：导出交付结果为 JSON 报告
python -m ctpipe export --output delivery_20260609/report.json

# clean：预览清理内容
python -m ctpipe clean --dry-run

# clean：清理 runs + 缓存 + 旧批次
python -m ctpipe clean --cache --old-deliveries

# clean：仅清理缓存（跳过 runs）
python -m ctpipe clean --no-runs --cache
```

---

## 幂等性与断点续跑

Pipeline 通过 `pipeline_state.json` 记录每个任务在每个阶段的完成状态。每个阶段在执行前检查 `is_done()`，已完成的任务会被跳过。

这意味着：
- 中途失败后直接重新运行 `all`，只会执行未完成的部分
- 单独重跑某个阶段不会覆盖已完成的任务
- prepare 阶段额外检查运行目录是否仍存在，清理后重跑会自动重新克隆
- 如需强制重跑，使用 `python -m ctpipe reset --tasks CT-0001 --stages run collect` 重置指定阶段

---

## Passrate 阈值规则

```text
qwen passrate  < 0.7
claude passrate > 0.7
claude passrate > qwen passrate
(claude passrate - qwen passrate) / qwen passrate > 25%
```

Passrate 计算公式：

```text
passrate = sum(score_i × weight_i) / sum(points_i × weight_i)
```

分母按实际选中的维度数量、points 和 weight 动态计算。weight 从评分模板 TOML 中读取（与架构边界/安全合规相关的维度通常为 2.0，其余为 1.0）。

---

## 并发与性能调优

`tasks.toml` 中的 `max_parallel` 控制同时运行的 `claude -p` 进程数上限。每个任务内部 Qwen 和 Claude 并行执行（各占 1 个进程槽位），所以 `max_parallel=3` 意味着最多 3 个 `claude -p` 进程同时运行，通常可推进 1-2 个任务。

| max_parallel | 并发进程 | 并发任务数 | 建议 |
|:---:|:---:|:---:|:---:|
| 3 | 3 | 1-2 | 8GB 内存 |
| 6 | 6 | 3 | 16GB 内存 |

瓶颈通常在 API 端限流而非本地资源。

### 大批量任务（>30 条）运行注意事项

当任务数较多时（例如 45-60 条），`run` 阶段耗时 6-8 小时。以下是可能遇到的问题和应对策略：

| 风险 | 表现 | 应对 |
|------|------|------|
| API 限流 | 某些 turn 返回 429/5xx，标记为 `partial` | 跑完后用 `reset` 重跑失败任务 |
| 网络中断 | 进程 timeout（600s），标记 `partial` | 降低 `max_parallel` 或重跑 |
| 进程被杀 | Ctrl+C / 系统休眠 | 直接重跑 `all`，已完成的自动跳过 |
| Batch 等待 | 一个慢任务拖住整批 | 不影响正确性，仅浪费时间 |

**推荐分步执行**（而非一键 `all`）：

```powershell
# 1. 准备（快速，约 30 分钟）
python -m ctpipe prepare

# 2. 执行（耗时最长，可中断恢复）
python -m ctpipe run --turn-timeout 900 --total-timeout 2400

# 3. 检查 run 结果，决定是否重跑失败任务
python -m ctpipe check

# 4. 重跑失败任务（如有）
python -m ctpipe reset --tasks CT-0010 CT-0015 --stages run
python -m ctpipe run --tasks CT-0010 CT-0015

# 5. 后续阶段
python -m ctpipe collect
python -m ctpipe score
python -m ctpipe finalize
python -m ctpipe validate
```

**耗时参考**（以 45 条任务、`max_parallel=6` 为例）：

| 阶段 | 预估耗时 | 说明 |
|:---:|:---:|------|
| prepare | 30-60 min | 90 次 git clone |
| run | 6-8 小时 | 瓶颈阶段，90 个 run ÷ 6 并发 |
| collect | 1-2 min | 并行文件复制 |
| score | 30-60 min | 90 次 AI 评分 |
| finalize | <1 min | 纯计算 |

> **安全保证**：`pipeline_state.json` 保证已完成任务不会被重跑。中途中断后直接重新执行对应阶段即可继续，不会丢失已有进度。

---

## 安全加固

Pipeline 在执行过程中涉及子进程调用、外部仓库克隆、API 认证等操作，已实施以下安全防护措施：

### 输入验证

所有用户可控的标识符在使用前都经过严格验证，防止路径遍历和注入攻击：

| 验证函数 | 校验对象 | 防御目标 |
|----------|----------|----------|
| `validate_task_id()` | 任务 ID（`CT-xxxx`） | 必须匹配 `^CT-\d{4}$`，阻止 `../` 等注入 |
| `validate_delivery_date()` | 交付日期（`YYYYMMDD`） | 必须为 8 位数字，阻止路径遍历 |
| `validate_session_id()` | Session ID | 阻止 `..`、`/`、`\`、`\0` 等危险字符 |
| `validate_path_component()` | 文件路径组件 | 阻止路径遍历 + Windows 保留设备名（CON、NUL 等） |
| `is_safe_clone_url()` | Git clone URL | 仅允许 `https://`、`http://`、`git://` 协议，阻止 `file://`、`ssh://` 等本地协议 |

### 子进程沙箱化

克隆的外部仓库可能包含恶意配置文件。在以 `--dangerously-skip-permissions` 执行前，Pipeline 会自动清理：

- 移除仓库中的 `.claude/` 目录（可能包含恶意 hooks 或 settings）
- 移除 `CLAUDE.md`、`claude.md`、`.claudeignore`（可能注入恶意指令）
- 重建受信任的 `.claude/settings.local.json`（来自 Pipeline 内部配置）

### 环境变量隔离

`run` 阶段的子进程使用白名单环境变量，而非继承完整的 `os.environ`：

- 仅传递必要变量：`ANTHROPIC_*`、`PATH`、`HOME`、`TEMP`、`SystemRoot`、`ComSpec` 等
- `GIT_*` 限制为安全前缀（`GIT_AUTHOR_`、`GIT_COMMITTER_`、`GIT_DIR`、`GIT_WORK_TREE`）
- API Token 不会泄露到子进程的额外环境变量中

### 进程列表安全

- **Prompt 传递**：通过 stdin 传入 `claude -p`，而非 CLI 参数（`--prompt`），防止 prompt 内容出现在 `ps` / 任务管理器的命令行中
- **API Token 传递**：`github_search.py` 中的 curl 调用通过 `--config -` 从 stdin 传入 Authorization header，防止 token 出现在进程列表中

### 数据完整性

- **TOML 注入防护**：`gen.py` 中写入 TOML 文件时，所有字符串值经过 `_escape_toml_basic_string()` 转义，防止 AI 生成的内容注入恶意 TOML 结构
- **线程安全状态管理**：`PipelineState` 使用 `threading.RLock()` 保证并发读写安全，`save()` 使用原子写入（先写 `.tmp` 再 rename）
- **文件锁定**：`_save_used_repo()` 使用 `filelock.FileLock` 防止并发写入冲突

### 文件系统防护

- **符号链接防护**：`project_scan.py` 在扫描项目时跳过符号链接，防止通过 symlink 逃逸读取项目外文件
- **安全删除**：`clean.py` 在执行 `shutil.rmtree` 前验证目标路径必须在 `runs_root` 或 `delivery_dir` 之下
- **Clone 目录验证**：`--clone-dir` 参数经过 `_validate_runs_root()` 校验，确保是合法的绝对路径
- **导出路径验证**：`export` 命令的 `--output` 路径验证必须在项目目录范围内

### 网络安全

- **Retry-After 上限**：HTTP 429 响应的 `Retry-After` 值限制为最大 120 秒，防止恶意服务器通过超大值造成 DoS
- **Clone URL 协议限制**：仅允许 `https://`、`http://`、`git://` 协议的 clone URL，阻止 `file://`、`ssh://` 等可能的本地文件读取或 SSRF 攻击

---

## 重要约束

- 不要提交 `.env`、API 密钥或任何真实 token
- 不要手动编辑导出的 trajectory JSONL 文件
- Session ID 必须从 trajectory 内容中读取
- Qwen 和 Claude 可以使用不同的 follow-up，但必须基于相同的代码库和任务主题
- Python 3.11+ 是必需的（`tomllib` 依赖）

---

## 相关文档

- [CLAUDE.md](./CLAUDE.md) — Claude Code 的项目指令
- [docs/评分规范.md](./docs/评分规范.md) — 评分规则和标准详细说明
- [docs/评分文件规范.md](./docs/评分文件规范.md) — 评分 TOML 文件格式规范
- [docs/analyze_prompt_template.md](./docs/analyze_prompt_template.md) — `--analyze` 模式的提示词模板
- [rubrics_templates/README.md](./rubrics_templates/README.md) — 评分模板说明
