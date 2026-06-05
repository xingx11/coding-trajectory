# Coding Trajectory 基准数据生产工作区

本仓库是一个数据生产工作区，用于生成 Coding Trajectory 基准测试样本，比较 Qwen 和 Claude 在真实本地编码任务中的表现。

每条样本包含：一对 Qwen/Claude 的 trajectory JSONL、一对评分文件、任务元数据，以及写入 submission.csv 的汇总行。

---

## 目录结构

```text
.
├─ ctpipe/                    # 自动化 pipeline 源码
│  ├─ cli.py                  # 命令行入口
│  ├─ config.py               # tasks.toml + .env 配置加载
│  ├─ distribution.py         # 225 行题目分布表 + 加权采样
│  ├─ github_search.py        # GitHub REST API 项目搜索 + clone
│  ├─ gen.py                  # AI 全自动任务生成
│  ├─ prepare.py              # 项目克隆 + 交付目录骨架
│  ├─ run.py                  # claude -p 多轮执行
│  ├─ collect.py              # trajectory JSONL 收集
│  ├─ score.py                # AI 自动评分
│  ├─ finalize.py             # passrate 计算 + submission.csv
│  ├─ validate.py             # 完整性校验
│  ├─ state.py                # JSON 状态管理（幂等重跑）
│  ├─ trajectory.py           # JSONL 解析工具
│  ├─ toml_utils.py           # TOML 评分文件读写
│  └─ project_hash.py         # Windows 路径 → Claude 项目 hash
├─ docs/                      # 参考文档、模板、示例
│  ├─ examples/               #   参考 JSONL 和 TOML 示例
│  └─ ...
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
max_parallel = 3                   # 并发任务数
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

---

## 完整开发流程

从零开始到产出一个可交付批次，完整流程分为三大阶段：**任务准备（可全自动） → 自动化执行 → 人工审核**。

### 第一阶段：任务准备

任务准备支持两种方式：**全自动生成**（推荐）和**人工编写**。

#### 方式一：全自动生成（推荐）

使用 `gen` 子命令，基于需求文档中的 225 行题目分布表自动完成全部准备工作：

```powershell
# 按分布自动生成 45 条任务（每个项目生成 3 条，共克隆 15 个仓库）
python -m ctpipe gen --count 45 --per-project 3

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

# 预览模式（不 clone、不写文件、不调用 AI）
python -m ctpipe gen --count 45 --per-project 3 --dry-run
```

`gen` 的完整流程：

1. **加权采样**：从 225 行分布表中按权重随机选取 N 个 `(task_type, domain, language)` 组合
2. **分组**（`--per-project > 1` 时）：将 N 个 slot 按每组 M 个分组，每组共享一个项目
3. **搜索项目**：根据 domain 和 language，通过 GitHub REST API 搜索合适的开源项目（stars ≥ 50、非 fork、非 archived、< 100MB）
4. **克隆项目**：`git clone --depth 1 --filter=blob:none` 到指定目录
5. **扫描项目**：提取 README、目录树、依赖文件，生成 ~1.5K 字符的项目概要
6. **AI 发现任务**：
   - 单任务模式（`per_project=1`）：两阶段 — idea 生成 + expand（共 2 次 API 调用）
   - 批量模式（`per_project>1`）：先批量生成 M 个 idea（1 次调用），再逐个 expand（M 次调用），共 M+1 次
7. **写入配置**：
   - 自动递增 task_id（`CT-xxxx`）
   - 生成 rubric 模板写入 `rubrics_templates/qwen/` 和 `rubrics_templates/claude/`
   - 格式化 `[[task]]` 条目追加到 `tasks.toml`

**性能对比**（以 45 条任务为例）：

| 模式 | 仓库数 | API 调用数 | 预估耗时 |
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

评分模板包含 7 个维度，每个维度 0-5 分，满分 35 分：

| 维度 | 含义 |
|------|------|
| user_experience_and_interaction | 用户体验与交互质量 |
| task_planning_and_execution_control | 任务规划与执行控制 |
| semantic_understanding_and_logical_reasoning | 语义理解与逻辑推理 |
| instruction_compliance_and_constraint_adherence | 指令遵守与约束合规 |
| engineering_quality_and_completeness | 工程质量与完整度 |
| delivery_completeness_and_usability | 交付完整性与可用性 |
| architecture_boundaries_and_security_compliance | 架构边界与安全合规 |

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
4. 每轮有 `turn_timeout`（默认 600s），整体有 `total_timeout`（默认 1800s）
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
1. 从 trajectory JSONL 提取精简文本（用户消息 + 助手回复 + 工具调用摘要，上限 100K 字符）
2. 将评分模板和 trajectory 文本拼接成 prompt，通过 `claude -p --bare` 调用 AI 评分
3. AI 返回填好 score 和 rationale 的 TOML
4. 解析返回的 TOML 并覆盖写入交付目录的评分文件
5. 解析失败时保存为 `.draft.txt` 供人工处理

#### Stage 5：Finalize — 汇总提交

```powershell
python -m ctpipe finalize
```

做了什么：
1. 读取所有评分文件，计算 passrate（`sum(score × weight) / sum(points × weight)`）
2. 检查阈值：qwen < 0.7、claude >= 0.71、claude > qwen
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

---

### 第三阶段：人工审核（人工）

Pipeline 产出的是 AI 初评结果，需要人工复核：

1. **审核评分**：打开 `delivery_YYYYMMDD/scores/` 下的 `.quality.toml` 文件，检查 AI 给出的 score 和 rationale 是否合理，手动修正
2. **检查 passrate 阈值**：确认每条数据满足 `qwen < 0.7`、`claude >= 0.71`、`claude > qwen`
3. **补充 metadata**：在 `metadata/CT-xxxx.md` 中记录任务背景和特殊说明
4. **重新 finalize + validate**：修正评分后重新运行以更新 submission.csv

```powershell
python -m ctpipe finalize
python -m ctpipe validate
```

---

## 常用命令参考

| 命令 | 说明 |
|------|------|
| `python -m ctpipe gen --count 45 --per-project 3` | 全自动生成 45 条任务（15 个仓库，每个 3 条） |
| `python -m ctpipe gen --count 45` | 全自动生成 45 条任务（45 个仓库，每个 1 条） |
| `python -m ctpipe gen --count 45 --per-project 3 --source gitee` | 从 Gitee 搜索项目（国内免代理） |
| `python -m ctpipe all` | 运行全流程（prepare→run→collect→score→finalize→validate） |
| `python -m ctpipe prepare` | 仅克隆项目 + 创建交付骨架 |
| `python -m ctpipe run --tasks CT-0001 CT-0002` | 运行指定任务 |
| `python -m ctpipe run --models qwen` | 仅运行 Qwen |
| `python -m ctpipe collect` | 收集 trajectory JSONL |
| `python -m ctpipe score` | AI 自动评分 |
| `python -m ctpipe finalize` | 计算 passrate + 生成 submission.csv |
| `python -m ctpipe validate` | 校验交付完整性 |
| `python -m ctpipe validate --models qwen` | 仅校验 Qwen 侧数据 |
| `python -m ctpipe reset --tasks CT-0001 --stages run collect` | 重置指定任务的指定阶段状态 |
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

# gen：预览模式（不 clone、不写文件、不调用 AI）
python -m ctpipe gen --count 45 --per-project 3 --dry-run
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
claude passrate >= 0.71
claude passrate > qwen passrate
```

Passrate 计算公式：

```text
passrate = sum(score_i × weight_i) / sum(points_i × weight_i)
```

当所有维度 weight=1.0、points=5 时，简化为 `sum(7个分数) / 35`。

---

## 并发与性能调优

`tasks.toml` 中的 `max_parallel` 控制同时运行的 `claude -p` 进程数上限。每个任务内部 Qwen 和 Claude 并行执行（各占 1 个进程槽位），所以 `max_parallel=3` 意味着最多 3 个 `claude -p` 进程同时运行，通常可推进 1-2 个任务。

| max_parallel | 并发进程 | 并发任务数 | 建议 |
|:---:|:---:|:---:|:---:|
| 3 | 3 | 1-2 | 8GB 内存 |
| 6 | 6 | 3 | 16GB 内存 |

瓶颈通常在 API 端限流而非本地资源。

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
- [rubrics_templates/README.md](./rubrics_templates/README.md) — 评分模板说明
