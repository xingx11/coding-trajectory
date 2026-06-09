# ctpipe collect 阶段：salvage 与 force 恢复机制

## 1 概述

`ctpipe collect` 是流水线的第三阶段，负责将 `run` 阶段产出的 JSONL 轨迹文件（位于 `~/.claude/projects/<hash>/` 下）查找、校验并拷贝到交付目录 `delivery_YYYYMMDD/trajectories/{model}/` 中。

当 `run` 阶段被中断（进程被杀、网络超时、Ctrl+C 等），JSONL 文件可能已部分写入磁盘，但 `pipeline_state.json` 中的运行状态并非 `"done"` 或 `"partial"`。此时正常的 collect 流程会跳过这些任务，导致已有数据丢失。

**salvage**（打捞）和 **force**（强制）是 collect 阶段提供的两种恢复机制，用于在异常场景下尽可能回收数据。两者通过 `collect.py`、`trajectory.py`、`state.py`、`project_hash.py`、`retry.py`、`cli.py` 六个模块协作完成。

---

## 2 模块职责总览

| 模块 | 文件 | 在恢复机制中的角色 |
|------|------|--------------------|
| **cli** | `ctpipe/cli.py` | 解析 `--no-salvage` / `--force` 命令行标志，路由到 `collect_all()` |
| **collect** | `ctpipe/collect.py` | 核心逻辑：状态判断、校验规则选择、分支控制、状态写入 |
| **trajectory** | `ctpipe/trajectory.py` | JSONL 文件定位（`find_trajectory_for_run`）与解析（`parse_trajectory`） |
| **state** | `ctpipe/state.py` | 线程安全的 JSON 状态账本，读写各阶段结果 |
| **project_hash** | `ctpipe/project_hash.py` | 将 `run_dir` 路径映射为 Claude 项目哈希目录 |
| **retry** | `ctpipe/retry.py` | 自动重试引擎，级联重置下游阶段，隐式利用 salvage |

---

## 3 流水线状态机

每个 task/model 组合在各阶段的状态记录在 `delivery_YYYYMMDD/pipeline_state.json`，由 `PipelineState` 类管理。

### 3.1 状态值

| 状态 | 含义 |
|------|------|
| `""` | 待执行（尚未写入状态） |
| `"running"` | run 阶段进行中 |
| `"done"` | 正常完成 |
| `"partial"` | 部分完成（salvage 成功 或 run 部分完成） |
| `"failed"` | 失败 |
| `"error"` | 异常 |
| `"timeout"` | 超时 |
| `"skipped"` | 跳过（salvage 未找到 JSONL） |
| `"draft"` | 草稿 |
| `"permanently_failed"` | 超过最大重试次数，永久失败 |

### 3.2 核心 API

```python
state.get(task_id, stage, model)       # 读取状态
state.set(task_id, stage, model, ...)  # 写入状态（原子写入：先 .tmp 再 replace）
state.is_done(task_id, stage, model)   # 仅当 status == "done" 时返回 True
state.reset(task_id, stage, model)     # 删除条目
state.batch()                          # 上下文管理器，批量操作结束后一次性落盘
```

---

## 4 正常 collect 流程

正常流程由 `collect_single()` 函数执行：

```
1. 读取 run 阶段状态 → run_status
2. 仅当 run_status ∈ {"done", "partial"} 时继续；否则跳过，返回 False
3. 从状态中读取 session_id 和 start_time
   - start_time 缺失 → _infer_start_time()：从 .claude/ 目录文件 mtime 推断
   - session_id 缺失 → _infer_session_id()：扫描项目哈希目录最新 JSONL 提取
4. 调用 find_trajectory_for_run(run_dir, start_time, session_id) 定位 JSONL
5. 调用 parse_trajectory() 解析 JSONL → TrajectoryInfo
6. 校验：
   - provider 匹配（qwen vs claude）
   - 行数 ≥ 10
   - models 集合非空
   - session_id 匹配
7. 拷贝到 delivery_dir/trajectories/{model}/{task_id}.jsonl
8. 写入状态：status="done"，附带 session_id、line_count 等元数据
```

---

## 5 salvage（打捞）机制

### 5.1 触发条件

- `salvage=True`（**默认开启**，可通过 `--no-salvage` 关闭）
- 且 `run_status ∈ {"running", "error", "failed", "timeout"}`

```python
# collect.py:14
_SALVAGEABLE_STATUSES = ("running", "error", "failed", "timeout")
```

### 5.2 放宽的校验规则

salvage 模式通过设置 `is_salvage = True` 来放宽四项校验：

| 校验项 | 正常模式 | salvage 模式 |
|--------|----------|-------------|
| 最低行数 | ≥ 10 (`MIN_TRAJECTORY_LINES`) | ≥ 3 (`MIN_SALVAGE_LINES`) |
| models 集合 | 必须非空 | 跳过检查 |
| session_id 不匹配 | → `status="failed"` | → 输出 WARN，继续执行 |
| 未找到 JSONL | → `status="failed"` | → `status="skipped"`，`recovery=True` |

### 5.3 输出状态

```python
state.set(task_id, "collect", model=model_name,
    status="partial",                  # 始终为 "partial"，不会是 "done"
    recovery=True,                     # 标记为恢复操作
    salvaged=True,                     # 标记为 salvage（区别于 force）
    forced=False,
    run_status_at_collect=run_status,  # 保留原始 run 状态
    jsonl_source=str(jsonl_path),
    session_id=info.session_id,
    model_detected=info.detected_provider,
    line_count=info.line_count,
)
```

### 5.4 错误路径

即使 salvage 失败，`recovery=True` 仍然会写入状态，以便下游工具识别这是一次恢复尝试：

| 场景 | status | recovery | 备注 |
|------|--------|----------|------|
| run 目录不存在 | `"failed"` | `True` | `error="run dir not found"` |
| 未找到 JSONL | `"skipped"` | `True` | `error="salvage: no JSONL"` |
| provider 不匹配 | `"failed"` | `True` | 与正常模式行为一致 |
| 行数 < 3 | `"failed"` | `True` | salvage 最低阈值仍不满足 |

### 5.5 典型使用场景

```bash
# run 阶段被 Ctrl+C 中断后，直接 collect 即可自动 salvage（默认开启）
python -m ctpipe collect

# 明确禁用 salvage，只收集正常完成的任务
python -m ctpipe collect --no-salvage
```

---

## 6 force（强制）机制

### 6.1 触发条件

- `force=True`（通过 `--force` 命令行标志激活）

### 6.2 行为覆盖

force 在两个层面绕过限制：

**层面一：`collect_all()` 层级**

```python
# collect.py:238
if not force and state.is_done(task.id, "collect", model_name):
    print(f"[{task.id}/{model_name}] collect already done, skipping")
    return
```

当 `force=True` 时，即使 collect 状态已经是 `"done"`，也会重新执行收集。

**层面二：`collect_single()` 层级**

```python
# collect.py:103-105
if force:
    start_time = 0.0    # 匹配所有时间的文件
    session_id = ""      # 不做 session 过滤
```

`find_trajectory_for_run(run_dir, 0.0, None)` 在这种情况下直接返回**项目哈希目录下 mtime 最新的 JSONL 文件**。

### 6.3 与 salvage 的交互

force 不总是等同于 salvage。是否进入 salvage 语义取决于 run 状态：

```python
if force:
    if run_status not in ("done", "partial"):
        is_salvage = True   # run 异常 → 叠加 salvage 语义（放宽校验）
    # run 正常 → is_salvage 保持 False（使用严格校验）
```

| run 状态 | force 行为 |
|----------|-----------|
| `"done"` / `"partial"` | 重新收集，**严格校验**，结果 `status="done"` |
| `"running"` / `"error"` / `"failed"` / `"timeout"` | 重新收集，**放宽校验**（salvage 语义），结果 `status="partial"` |

### 6.4 输出状态

```python
state.set(task_id, "collect", model=model_name,
    status="partial" if is_salvage else "done",
    recovery=True,                   # force 时始终为 True
    salvaged=is_salvage,             # run 异常时为 True
    forced=True,                     # 标记为 force 操作
    ...
)
```

### 6.5 典型使用场景

```bash
# 状态损坏，start_time/session_id 完全不对，强制选最新的 JSONL
python -m ctpipe collect --force --tasks CT-0001

# 对已经 collect 成功的任务重新收集（会覆盖旧文件）
python -m ctpipe collect --force --tasks CT-0001
```

---

## 7 `collect_single()` 完整决策树

> 源码位置：`ctpipe/collect.py` 第 73–227 行。
> 常量：`_SALVAGEABLE_STATUSES = ("running", "error", "failed", "timeout")`
> 每个叶节点标注 `→ return` 和写入的 `status`。

### 阶段一：模式判定（第 85–97 行）

确定 `is_salvage` 的值，并决定函数是否继续执行。

```
collect_single(task, model, config, state, salvage, force)
│
│  读取 run_info = state.get(task, "run", model)
│  提取 run_status = run_info["status"]  (默认 "")
│  初始化 is_salvage = False
│
│─ ❶ force == True ?  ─────────────────────────────────────── 【force 路径】
│   │
│   ├─ YES
│   │   │
│   │   ├─ run_status ∉ {"done", "partial"} ?
│   │   │   ├─ YES → is_salvage = True    （force + 异常 run → 叠加 salvage 语义）
│   │   │   └─ NO  → is_salvage = False   （force + 正常 run → 仅绕过参数验证）
│   │   │
│   │   └─ ▶ 继续执行 ──────────────────────────────────────────────────────┐
│   │                                                                        │
│─ ❷ elif run_status ∈ {"done", "partial"} ?  ────────────── 【正常路径】    │
│   │                                                                        │
│   ├─ YES                                                                   │
│   │   │  is_salvage = False                                                │
│   │   └─ ▶ 继续执行 ──────────────────────────────────────────────────────┤
│   │                                                                        │
│─ ❸ elif salvage == True                                                    │
│      AND run_status ∈ _SALVAGEABLE_STATUSES ?  ──────────── 【salvage 路径】│
│   │                                                                        │
│   ├─ YES                                                                   │
│   │   │  is_salvage = True                                                 │
│   │   └─ ▶ 继续执行 ──────────────────────────────────────────────────────┤
│   │                                                                        │
│─ ❹ else （兜底：不满足上述任何条件）                                        │
│   │                                                                        │
│   └─ ✘ return False                                                        │
│      状态写入: 无（不写任何状态）                                            │
│      日志: "run not done (status=...), skipping collect"                    │
│                                                                            │
═══════════════════════════════════════════════════════════════════════════════
                                                                             │
```

**三条路径触发条件总结：**

| 路径 | 触发条件 | `is_salvage` | `force` |
|------|---------|-------------|---------|
| 正常 | `run_status ∈ {"done","partial"}` 且 `force=False` | `False` | `False` |
| salvage | `run_status ∈ _SALVAGEABLE` 且 `salvage=True` 且 `force=False` | `True` | `False` |
| force | `force=True`，run_status 为任意值 | 取决于 run_status | `True` |

---

### 阶段二：参数解析（第 99–112 行）

根据阶段一确定的模式，准备 `run_dir`、`start_time`、`session_id` 三个关键参数。

```
（从阶段一继续）◀─────────────────────────────────────────────────────────────┘
│
│  session_id = run_info.get("session_id", "")
│  prepare_info = state.get(task, "prepare")
│  run_dir = Path(prepare_info.get(f"{model}_dir", ""))
│
├─ force == True ?  ──────────────────────────────────────── 【force 参数】
│   │
│   ├─ YES
│   │   │  start_time = 0.0            ← 匹配所有时间的 JSONL
│   │   │  session_id = ""             ← 不做 session 过滤
│   │   │  （跳过所有推断逻辑）
│   │   └─ ▶ 进入校验链
│   │
│   └─ NO ──────────────────────────────────── 【正常 / salvage 参数】
│       │
│       │  start_time = run_info.get("start_time", None)
│       │
│       ├─ start_time is None OR start_time == 0 ?
│       │   ├─ YES → start_time = _infer_start_time(run_dir, ...)
│       │   │         ├─ .claude/ 目录存在 → min(文件 mtime)
│       │   │         └─ .claude/ 不存在   → 0.0（epoch 兜底）
│       │   └─ NO  → 保持原值
│       │
│       ├─ session_id 为空 AND run_dir.is_dir() ?
│       │   ├─ YES → session_id = _infer_session_id(run_dir, start_time, ...)
│       │   │         ├─ 项目哈希目录有 JSONL → 读取前 30 行提取 sessionId
│       │   │         └─ 无可用 JSONL        → ""（空字符串）
│       │   └─ NO  → 保持原值
│       │
│       └─ ▶ 进入校验链
```

---

### 阶段三：校验链（第 114–196 行）

5 个顺序检查点，任一失败即提前返回。每个检查点都受 `is_salvage` 影响。

```
（从阶段二继续）
│
│
│══ 检查点 A：run_dir 是否存在（第 114 行）══════════════════════════════════
│
│   条件: run_dir.is_dir()
│
├── FALSE ─────────────────────────────────────────────────────────────────
│   │
│   │  state.set(status="failed",
│   │            error="run dir not found",
│   │            recovery = is_salvage or force)
│   │
│   │  ✘ return False
│   │
│   │  ┌─────────────────────────────────────────────────────────────────┐
│   │  │ recovery 的值:                                                  │
│   │  │   正常路径  → False                                             │
│   │  │   salvage   → True                                              │
│   │  │   force     → True                                              │
│   │  └─────────────────────────────────────────────────────────────────┘
│
├── TRUE → 继续
│
│
│══ 检查点 B：JSONL 文件是否存在（第 123–135 行）════════════════════════════
│
│   jsonl_path = find_trajectory_for_run(run_dir, start_time, session_id or None)
│
│   条件: jsonl_path is not None
│
├── FALSE
│   │
│   ├─ is_salvage == True ?
│   │   │
│   │   ├─ YES
│   │   │   │  state.set(status="skipped",
│   │   │   │            error="salvage: no JSONL",
│   │   │   │            recovery=True)
│   │   │   │
│   │   │   │  ✘ return False
│   │   │   │  日志: "salvage: no JSONL found, nothing to recover"
│   │   │
│   │   └─ NO （正常 / force+正常run）
│   │       │  state.set(status="failed",
│   │       │            error="no JSONL found")
│   │       │            ↑ 注意: 无 recovery 字段
│   │       │
│   │       │  ✘ return False
│   │       │  日志: "ERROR: no JSONL found for run"
│   │
├── TRUE → 继续
│
│
│══ 检查点 C：provider 是否匹配（第 137–151 行）════════════════════════════
│
│   info = parse_trajectory(jsonl_path)
│
│   条件: info.detected_provider ∈ {model_name, "unknown"}
│
├── FALSE （provider 不匹配，如期望 qwen 但检测到 claude）
│   │
│   │  state.set(status="failed",
│   │            error=f"provider mismatch: detected {info.detected_provider}",
│   │            recovery = is_salvage or force)
│   │
│   │  ✘ return False
│   │
│   │  ┌─────────────────────────────────────────────────────────────────┐
│   │  │ 此检查对三条路径行为一致，均严格校验。                            │
│   │  │ 区别仅在 recovery 字段:                                         │
│   │  │   正常路径  → recovery=False                                    │
│   │  │   salvage   → recovery=True                                     │
│   │  │   force     → recovery=True                                     │
│   │  └─────────────────────────────────────────────────────────────────┘
│
├── TRUE → 继续
│
│
│══ 检查点 D：行数 + models 完整性（第 153–175 行）═════════════════════════
│
│   MIN_TRAJECTORY_LINES = 10
│   MIN_SALVAGE_LINES    = 3
│   min_lines = MIN_SALVAGE_LINES if is_salvage else MIN_TRAJECTORY_LINES
│
│   复合条件: info.line_count < min_lines
│             OR
│             (not is_salvage AND not info.models)
│
├── TRUE （未通过完整性检查）
│   │
│   ├─ is_salvage == True AND info.line_count > 0 ?
│   │   │
│   │   ├─ YES ──────────────────────────────────── 【salvage 特有：降级通过】
│   │   │   │  不 return！跳出此检查点，继续执行
│   │   │   │  日志: "salvage: trajectory too short (lines=N), collecting as partial"
│   │   │   │
│   │   │   └─ ▶ 进入检查点 E
│   │   │
│   │   └─ NO （正常失败 或 salvage 但 line_count==0）
│   │       │  state.set(status="failed",
│   │       │            error=f"trajectory incomplete: lines={N}, models={M}",
│   │       │            recovery = is_salvage or force)
│   │       │
│   │       │  ✘ return False
│   │
├── FALSE （完整性检查通过）→ 继续
│
│
│══ 检查点 E：session_id 是否匹配（第 177–196 行）═════════════════════════
│
│   前提: session_id 非空（force 路径中 session_id=""，直接跳过此检查）
│
│   条件: session_id == "" OR info.session_id == session_id
│
├── FALSE （session_id 不匹配）
│   │
│   ├─ is_salvage == True ?
│   │   │
│   │   ├─ YES ──────────────────────────────────── 【salvage 特有：降级通过】
│   │   │   │  不 return！输出 WARN，继续执行
│   │   │   │  日志: "WARN: session_id mismatch — expected X, got Y (continuing salvage)"
│   │   │   │
│   │   │   └─ ▶ 进入阶段四（成功路径）
│   │   │
│   │   └─ NO （正常 / force+正常run）
│   │       │  state.set(status="failed",
│   │       │            error=f"session_id mismatch: expected {X}, got {Y}",
│   │       │            recovery = is_salvage or force)
│   │       │
│   │       │  ✘ return False
│   │
├── TRUE → 继续
│
```

---

### 阶段四：成功路径（第 198–227 行）

所有检查通过后执行拷贝并写入最终状态。

```
（从检查点 E 继续）
│
│  dest_dir  = config.delivery_dir / "trajectories" / model_name
│  dest_path = dest_dir / f"{task.id}.jsonl"
│  shutil.copy2(jsonl_path, dest_path)
│
│  is_recovery  = is_salvage or force
│  final_status = "partial" if is_salvage else "done"
│  label        = "Salvaged" if is_salvage else ("Forced" if force else "Copied")
│
│  state.set(task, "collect", model,
│      status             = final_status,     ← 见下表
│      recovery           = is_recovery,
│      salvaged           = is_salvage,
│      forced             = force,
│      run_status_at_collect = run_status,
│      jsonl_source       = str(jsonl_path),
│      jsonl_path         = 相对路径,
│      session_id         = info.session_id,
│      model_detected     = info.detected_provider,
│      line_count         = info.line_count,
│  )
│
│  ✔ return True
│
```

**成功路径输出矩阵：**

| 路径 | `is_salvage` | `force` | `status` | `recovery` | `salvaged` | `forced` | `label` |
|------|-------------|---------|----------|-----------|-----------|---------|---------|
| 正常 | `False` | `False` | `"done"` | `False` | `False` | `False` | `"Copied"` |
| salvage | `True` | `False` | `"partial"` | `True` | `True` | `False` | `"Salvaged"` |
| force + run 正常 | `False` | `True` | `"done"` | `True` | `False` | `True` | `"Forced"` |
| force + run 异常 | `True` | `True` | `"partial"` | `True` | `True` | `True` | `"Salvaged"` |

---

### 全景速览图

```
collect_single()
│
│  ┌─────────────────────────── 阶段一: 模式判定 ──────────────────────────┐
│  │                                                                       │
│  │  force?──YES──┬── run 异常? ──YES── is_salvage=T ──┐                  │
│  │               └── run 正常? ──YES── is_salvage=F ──┤                  │
│  │                                                     │                  │
│  │  run∈{done,partial}? ──YES── is_salvage=F ─────────┤                  │
│  │                                                     │                  │
│  │  salvage=T & run∈可打捞? ──YES── is_salvage=T ─────┤                  │
│  │                                                     │                  │
│  │  else ─────────────────── ✘ return False            │                  │
│  │                                                     │                  │
│  └─────────────────────────────────────────────────────┼──────────────────┘
│                                                        │
│  ┌──────────────── 阶段二: 参数解析 ──────────────────┤──────────────────┐
│  │                                                     │                  │
│  │  force?──YES── start_time=0.0, session_id=""  ──────┤─── ▶ 校验链     │
│  │                                                     │                  │
│  │  NO ── 从状态读取 / 推断 start_time 和 session_id ──┘                  │
│  │                                                                        │
│  └────────────────────────────────────────────────────────────────────────┘
│
│  ┌──────────────── 阶段三: 校验链（5 个检查点）──────────────────────────┐
│  │                                                                      │
│  │  [A] run_dir 存在?                                                   │
│  │      NO  → ✘ status="failed", recovery=is_salvage|force              │
│  │                                                                      │
│  │  [B] JSONL 找到?                                                     │
│  │      NO + is_salvage → ✘ status="skipped", recovery=True             │
│  │      NO + 非salvage  → ✘ status="failed"                             │
│  │                                                                      │
│  │  [C] provider 匹配?   ← 三条路径行为一致                              │
│  │      NO  → ✘ status="failed", recovery=is_salvage|force              │
│  │                                                                      │
│  │  [D] line_count ≥ min_lines 且 models 非空(仅非salvage)?              │
│  │      未通过 + is_salvage + lines>0 → ⚠ WARN，降级通过（继续）         │
│  │      未通过 + 其他 → ✘ status="failed", recovery=is_salvage|force    │
│  │                                                                      │
│  │  [E] session_id 匹配?                                                │
│  │      不匹配 + is_salvage → ⚠ WARN，降级通过（继续）                   │
│  │      不匹配 + 非salvage  → ✘ status="failed", recovery=is_salvage|F  │
│  │                                                                      │
│  └──────────────────────────────────────────────────────────────────────┘
│
│  ┌──────────────── 阶段四: 成功路径 ────────────────────────────────────┐
│  │                                                                      │
│  │  shutil.copy2(jsonl → delivery_dir)                                  │
│  │                                                                      │
│  │  status   = "partial" if is_salvage else "done"                      │
│  │  recovery = is_salvage or force                                      │
│  │  salvaged = is_salvage                                               │
│  │  forced   = force                                                    │
│  │                                                                      │
│  │  ✔ return True                                                       │
│  │                                                                      │
│  └──────────────────────────────────────────────────────────────────────┘
```

---

## 8 跨模块协作流程

### 8.1 完整数据流

```
                        cli.py
                  解析 --no-salvage / --force
                         │
                         ▼
                   collect_all()
              ┌── ThreadPoolExecutor ──┐
              │                        │
              ▼                        ▼
      collect_single()          collect_single()
        task=A, qwen             task=A, claude
              │
              ├── state.get(task, "run", model)
              │     └── state.py: 读取 run 阶段状态
              │
              ├── state.get(task, "prepare")
              │     └── state.py: 读取 run_dir 路径
              │
              ├── _infer_start_time()           ← start_time 缺失时
              │     └── 读取 run_dir/.claude/ 文件 mtime
              │
              ├── _infer_session_id()           ← session_id 缺失时
              │     ├── project_hash.py: run_dir → 哈希目录
              │     └── 扫描哈希目录最新 JSONL → 提取 sessionId
              │
              ├── find_trajectory_for_run(run_dir, start_time, session_id)
              │     ├── project_hash.py: run_dir → 哈希目录
              │     ├── 快速路径: {session_id}.jsonl + mtime > start_time
              │     ├── 文件名匹配: stem == session_id
              │     ├── 内容匹配: JSONL 内 sessionId 字段
              │     └── 兜底: mtime 最新的 JSONL（force 走此路径）
              │
              ├── parse_trajectory(jsonl_path)
              │     └── 解析 → TrajectoryInfo(session_id, models, line_count, ...)
              │
              ├── 校验（严格 or 放宽，取决于 is_salvage）
              │     ├── provider 匹配？
              │     ├── line_count ≥ min_lines？
              │     ├── models 非空？（仅正常模式）
              │     └── session_id 匹配？（salvage: WARN；正常: FAIL）
              │
              ├── shutil.copy2(jsonl_path, delivery_dir/...)
              │
              └── state.set(task, "collect", model, ...)
                    └── state.py: 原子写入结果状态
                         │
                         ▼
                pipeline_state.json
                         │
            ┌────────────┼────────────┐
            ▼            ▼            ▼
        score.py    finalize.py    retry.py
        (阶段 4)    (阶段 5)      (自动重试)
```

### 8.2 trajectory.py 的查找策略

`find_trajectory_for_run()` 是 collect 定位 JSONL 文件的核心函数。salvage 和 force 通过控制传入参数来改变查找行为：

```python
def find_trajectory_for_run(run_dir, start_time, expected_session_id=None):
    proj_dir = project_hash_dir(run_dir)      # project_hash.py 提供路径映射

    # 快速路径：直接用 session_id 作为文件名查找
    if expected_session_id:
        direct = proj_dir / f"{expected_session_id}.jsonl"
        if direct.is_file() and direct.stat().st_mtime > start_time:
            return direct

    # 收集候选：所有 mtime > start_time 的 .jsonl 文件
    candidates = [...]

    # 按 session_id 精确匹配（文件名或文件内容）
    if expected_session_id:
        for f in candidates:
            if f.stem == expected_session_id or 文件内含 sessionId:
                return f

    # 兜底：取 mtime 最新的文件
    return candidates[0]  # ← force 模式（start_time=0, session_id=None）总是走这里
```

| 模式 | start_time | session_id | 效果 |
|------|-----------|------------|------|
| 正常 | 从状态读取 | 从状态读取 | 精确匹配目标 JSONL |
| salvage | 从状态读取（可能推断） | 从状态读取（可能推断） | 精确匹配，但校验放宽 |
| force | `0.0` | `None` | 兜底路径，取最新 JSONL |

### 8.3 project_hash.py 的路径映射

Claude Code 将每个项目的轨迹文件存储在 `~/.claude/projects/<hash>/` 目录下，其中 `<hash>` 是项目工作目录路径的编码。`project_hash.py` 提供 `project_hash_dir(run_dir)` 函数，将 `run_dir` 路径映射为对应的哈希目录。

这是 collect 能定位到 JSONL 文件的前提——无论 run 在哪个目录执行，都能找到 Claude 写入的轨迹文件。

---

## 9 与 retry 引擎的协作

### 9.1 retry 如何触发 collect

`retry.py` 扫描 `pipeline_state.json`，找到 `status ∈ {"failed", "partial", "draft"}` 的条目，重置状态后重新执行对应阶段。

当 `collect` 阶段的某条目失败时：

1. retry 将该条目加入重试列表
2. 通过级联映射展开下游：`collect → [score, finalize]`
3. 重置所有相关条目的状态
4. 按阶段顺序重新执行：先 `collect`，再 `score`，最后 `finalize`

```python
# retry.py:28-34
_DOWNSTREAM = {
    "prepare": ["run", "collect", "score", "finalize"],
    "run":     ["collect", "score", "finalize"],
    "collect": ["score", "finalize"],
    "score":   ["finalize"],
    "finalize": [],
}
```

### 9.2 retry 调用 collect 时的参数

```python
# retry.py:195-196
elif stage == "collect":
    collect_all(config, task_ids, models)   # salvage=True（默认）, force=False（默认）
```

**关键设计**：retry 使用默认的 `salvage=True`，这意味着当 run 阶段重试后仍然异常（如超时），retry 中的 collect 会自动 salvage 已有数据，而不是直接标记为失败。

### 9.3 retry 的级联效应

当 `run` 阶段重试时，`collect` 和 `score` 会被级联重置并重新执行——即使它们之前是成功的。这确保了 collect 始终基于最新的 run 产出。

```
run 重试 → collect 重置 → score 重置 → finalize 重置
         (cascade)     (cascade)     (cascade)
```

### 9.4 超过最大重试次数

当某条目的 `retry_count` 达到 `max_retries` 时，retry 将其标记为 `"permanently_failed"`：

```python
state.set(entry.task_id, entry.stage, entry.model,
    status="permanently_failed",
    retry_count=retry_count,
    error=last_error or "exceeded max retries",
)
```

`permanently_failed` 不在 `_RETRYABLE_STATUSES` 中，因此不会被后续的 retry 轮次再次触发。

---

## 10 `recovery` / `salvaged` / `forced` 三标志语义详解

### 10.1 逻辑关系

三个标志并非独立——它们之间存在严格的蕴含关系：

```
salvaged=True  ──→  recovery=True     （salvage 是 recovery 的子集）
forced=True    ──→  recovery=True     （force 是 recovery 的子集）

recovery = salvaged or forced         （recovery 是两者的并集）
```

因此 `recovery=True, salvaged=False, forced=False` 这一组合**在代码中不可能出现**。

### 10.2 各标志含义

#### `recovery`（恢复标记）

**赋值逻辑**：`recovery = is_salvage or force`

**语义**：此条目不是通过正常路径产出的，而是经过某种恢复手段获得。下游消费者（`check.py`、`stats.py`、`validate.py`）通过此标志过滤"干净数据"与"恢复数据"。

**触发场景**：

| 场景 | recovery |
|------|----------|
| 正常 collect（run=done，校验全通过） | `False` |
| salvage 成功或失败 | `True` |
| force 重收集（无论 run 状态） | `True` |
| 正常模式下校验失败（provider 不匹配、session 不匹配等） | `False` |
| salvage/force 模式下校验失败 | `True` |

**关键细节**：即使恢复操作最终失败（写入 `status="failed"` 或 `"skipped"`），`recovery` 仍为 `True`。这允许下游工具区分"正常失败"和"恢复失败"——前者可能需要重新 run，后者可能说明数据已无法恢复。

#### `salvaged`（打捞标记）

**赋值逻辑**：`salvaged = is_salvage`

**语义**：此条目的 run 阶段未正常完成（状态属于 `_SALVAGEABLE_STATUSES`），collect 放宽了校验规则后成功收集。数据可能不完整（行数可能少于 10、models 可能为空、session_id 可能不匹配）。

**触发场景**：

| 场景 | salvaged |
|------|----------|
| run 异常 + 非 force 路径 + salvage=True | `True` |
| run 异常 + force 路径 | `True`（force 叠加 salvage 语义） |
| run 正常 + force 路径 | `False`（force 不引入 salvage） |
| 正常 collect | `False` |

**下游影响**：`status` 被设为 `"partial"` 而非 `"done"`。`retry.py` 的 `_RETRYABLE_STATUSES` 包含 `"partial"`，因此 salvaged 的条目在后续 retry 中会被重新执行——如果 run 被重做并成功完成，collect 会覆盖为完整的 `"done"` 数据。

#### `forced`（强制标记）

**赋值逻辑**：`forced = force`

**语义**：此条目通过 `--force` 参数收集，绕过了 `start_time` 和 `session_id` 的精确匹配，直接选取了项目哈希目录下 mtime 最新的 JSONL 文件。数据本身可能完全正确，但来源的确定性低于正常路径。

**触发场景**：

| 场景 | forced |
|------|--------|
| `--force` 命令行参数 | `True` |
| 正常 collect | `False` |
| salvage（不带 `--force`） | `False` |

**下游影响**：`forced=True` 本身不改变 `status`（仍由 `is_salvage` 决定是 `"done"` 还是 `"partial"`）。它更多是一个审计标记，让运维人员能追溯哪些数据是通过强制手段收集的。

### 10.3 完整组合矩阵

| 路径 | `recovery` | `salvaged` | `forced` | `status` | 含义 |
|------|-----------|-----------|---------|----------|------|
| 正常 | `False` | `False` | `False` | `"done"` | 正常产出，校验全部通过 |
| salvage | `True` | `True` | `False` | `"partial"` | 从异常 run 中打捞，校验放宽 |
| force + run 正常 | `True` | `False` | `True` | `"done"` | 强制重收集，run 本身正常，严格校验 |
| force + run 异常 | `True` | `True` | `True` | `"partial"` | 强制重收集 + run 异常，放宽校验 |
| 不可能 | `True` | `False` | `False` | — | 代码中无法产生此组合 |

### 10.4 与 `run_status_at_collect` 的配合

`run_status_at_collect` 始终记录 collect 执行时 run 阶段的原始状态，与三个布尔标志互为补充：

```python
# 三个布尔标志回答"how" —— 用什么手段收集的
recovery=True, salvaged=True, forced=False

# run_status_at_collect 回答"why" —— 为什么需要恢复
run_status_at_collect="timeout"     # run 超时了
run_status_at_collect="running"     # run 被中断了（Ctrl+C / 进程被杀）
run_status_at_collect="failed"      # run 失败了
```

下游工具可以通过 `run_status_at_collect` 进行更精细的分析——例如统计"超时导致的 salvage"和"失败导致的 salvage"各占多少比例。

### 10.5 下游消费者行为

| 消费者 | 对三标志的处理 |
|--------|---------------|
| `retry.py` | 不直接读取三标志；通过 `status="partial"` 触发重试，salvaged 条目自动进入重试队列 |
| `check.py` | 可读取 `recovery` 标志，对恢复数据放宽某些检查（如行数阈值） |
| `stats.py` | 可统计 `recovery`/`salvaged`/`forced` 的数量和比例 |
| `validate.py` | 检查交付完整性时，`status="partial"` 的条目可能被标记为需要关注 |

### 10.6 `state.set()` 的替换与合并语义

理解三标志的写入行为，需要先理解 `state.set()` 的两种模式：

```python
# state.py:67-83
def set(self, task_id, stage, model=None, **data):
    with self._lock:
        if "status" in data:
            # 模式一：替换 —— 整个 dict 被 data 覆盖
            task[stage][model] = data
        else:
            # 模式二：合并 —— data 中的字段追加到已有 dict
            task[stage][model] = {**task[stage].get(model, {}), **data}
```

**规则**：当 `**data` 中包含 `status` 键时，整个条目被**完全替换**；否则仅**合并**传入的字段。

`collect_single()` 的每次 `state.set()` 调用都包含 `status` 键，因此每次写入都是**完整替换**——不会残留旧字段。这意味着：

- 如果一个条目先被 salvage 写入 `recovery=True, salvaged=True`
- 后被 retry 重新执行，正常 collect 写入 `recovery=False, salvaged=False`
- 最终状态中**不会**残留之前的 `recovery=True`

这是保证 retry 后数据"干净"的关键机制。

---

## 11 线程安全与 `batch()` 延迟写入机制

### 11.1 `PipelineState` 的双层保护模型

`PipelineState` 对并发访问提供两层保护，分别解决不同问题：

```
┌─────────────────────────────────────────────────────────────────┐
│  第一层：threading.Lock（_lock）                                 │
│  ─────────────────────────────────                               │
│  保护对象：内存中的 _data 字典                                    │
│  保护方式：每个 get/set/is_done/reset 操作独立加锁                │
│  解决问题：多线程同时读写 _data 时的数据竞争                       │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │  第二层：batch() 上下文管理器（_batch_depth 计数器）          │ │
│  │  ────────────────────────────────────────────────            │ │
│  │  保护对象：磁盘上的 pipeline_state.json 文件                  │ │
│  │  保护方式：batch 期间跳过 save()，退出时一次性落盘             │ │
│  │  解决问题：频繁磁盘 I/O 的性能瓶颈                            │ │
│  └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

**两层各司其职**：`_lock` 保证内存一致性，`batch()` 控制磁盘写入时机。

### 11.2 `_lock` 的加锁粒度

`_lock` 是一个 `threading.Lock()`，在**每次 API 调用**时独立加锁和释放：

```python
# state.py — 每个方法都是 lock → operate → unlock 的模式

def get(self, task_id, stage, model=None):
    with self._lock:                          # ← 加锁
        return self._data.get(task_id, ...)   # ← 读取
                                              # ← 自动解锁

def set(self, task_id, stage, model=None, **data):
    with self._lock:                          # ← 加锁
        task[stage][model] = data             # ← 修改内存
        if self._batch_depth == 0:            # ← 检查 batch 状态
            self.save()                       # ← 可能写磁盘
                                              # ← 自动解锁

def is_done(self, task_id, stage, model=None):
    with self._lock:                          # ← 加锁
        return info.get("status") == "done"   # ← 读取
                                              # ← 自动解锁
```

**粒度特征**：锁保护的是单次读/写操作的原子性，而非业务逻辑的事务性。一个 `collect_single()` 调用中包含多次 `state.get()` 和一次 `state.set()`，这些操作之间**不是**原子的——其他线程可以穿插执行。

### 11.3 `batch()` 延迟写入机制

#### 工作原理

```python
# state.py:42-52
@contextmanager
def batch(self):
    with self._lock:
        self._batch_depth += 1       # ① 进入：深度 +1
    try:
        yield                        # ② 执行 with 块内的代码
    finally:
        with self._lock:
            self._batch_depth -= 1   # ③ 退出：深度 -1
            if self._batch_depth == 0:
                self.save()          # ④ 最外层 batch 退出时落盘
```

`_batch_depth` 是一个支持**嵌套**的计数器。当 `batch()` 嵌套调用时，只有最外层的 `batch()` 退出时才触发 `save()`：

```python
with state.batch():           # depth: 0 → 1
    with state.batch():       # depth: 1 → 2
        state.set(...)        # depth=2，不写磁盘
    # depth: 2 → 1，不写磁盘
    state.set(...)            # depth=1，不写磁盘
# depth: 1 → 0，写磁盘 ← 唯一一次 save()
```

#### `set()` 内部的 batch 感知

```python
# state.py:67-83
def set(self, task_id, stage, model=None, **data):
    with self._lock:
        # ... 修改内存 _data ...
        if self._batch_depth == 0:   # ← 关键判断
            self.save()              # ← 仅在无 batch 时立即写磁盘
```

当 `_batch_depth > 0` 时，`set()` 仍然修改内存中的 `_data`，但**跳过** `save()` 调用。这意味着：

- batch 期间，所有 `set()` 的修改**立即可见**于同一进程内的其他线程（通过 `get()` 读取）
- batch 期间，磁盘文件**不更新**
- batch 退出时，所有累积的修改**一次性**写入磁盘

#### `save()` 的原子写入

```python
# state.py:28-32
def save(self) -> None:
    self._path.parent.mkdir(parents=True, exist_ok=True)
    tmp = self._path.with_suffix(".tmp")
    tmp.write_text(json.dumps(self._data, ...), encoding="utf-8")
    tmp.replace(self._path)            # ← POSIX rename / Windows MoveFileEx
```

`save()` 采用"写临时文件 + 原子替换"模式：先写入 `.tmp` 文件，再通过 `replace()` 原子替换目标文件。这保证了磁盘上不会出现半写的 JSON 文件。

### 11.4 `collect_all()` 的并发执行模型

```python
# collect.py:230-255
def collect_all(config, task_ids=None, models=None, *, salvage=True, force=False):
    state = PipelineState(config.delivery_dir / "pipeline_state.json")
    tasks = select_delivery_tasks(config, task_ids)
    models = models or ["qwen", "claude"]

    def _collect_task_model(task, model_name):
        if not force and state.is_done(task.id, "collect", model_name):   # ① 读取
            return
        collect_single(task, model_name, config, state, ...)              # ② 处理+写入

    with state.batch():                                                    # ③ batch 开始
        with ThreadPoolExecutor(max_workers=config.max_parallel) as executor:
            futures = []
            for task in tasks:
                for model_name in models:
                    futures.append(executor.submit(_collect_task_model, task, model_name))
            for future in as_completed(futures):
                exc = future.exception()
                if exc:
                    print(f"  ERROR in collect: {exc}")
    # ④ batch 结束 → 一次性 save()
```

#### 执行时序

```
主线程                        线程池 worker-1              线程池 worker-2
  │                                │                            │
  ├─ state.batch() enter           │                            │
  │   depth: 0→1                   │                            │
  │                                │                            │
  ├─ submit(CT-0001/qwen) ────────→│                            │
  ├─ submit(CT-0001/claude) ───────┼───────────────────────────→│
  ├─ submit(CT-0002/qwen) ────────→│                            │
  │   ...                          │                            │
  │                                ├─ is_done(CT-0001,qwen)     │
  │                                │   └─ lock → read → unlock  │
  │                                ├─ collect_single(...)       │
  │                                │   ├─ state.get() [加锁]    │
  │                                │   ├─ find_trajectory()     │
  │                                │   ├─ parse_trajectory()    │
  │                                │   └─ state.set() [加锁]    │
  │                                │       └─ 修改 _data        │
  │                                │       └─ depth>0, 跳过save │
  │                                │                            ├─ is_done(CT-0001,claude)
  │                                │                            ├─ collect_single(...)
  │                                │                            │   └─ state.set() [加锁]
  │                                │                            │       └─ 修改 _data
  │                                │                            │       └─ depth>0, 跳过save
  │                                │                            │
  ├─ as_completed() 等待所有 ──────┴────────────────────────────┴──
  │
  ├─ state.batch() exit
  │   depth: 1→0
  │   save() ──→ 一次性将所有修改写入磁盘
  │
  └─ return
```

### 11.5 原子性分析

#### 单条目写入：原子 ✓

每个 `state.set()` 调用在 `_lock` 保护下完成内存修改，且每个 task/model 组合的 collect 结果是**一次** `set()` 写入（包含所有字段：status、recovery、salvaged 等）。不会出现"写了一半字段"的情况。

#### 跨条目写入：非事务，但隔离 ✓

不同 task/model 的 `set()` 调用在各自的线程中独立执行。由于每条 `set()` 操作的是 `_data` 中不同的 key path（`task_id → stage → model`），不存在写冲突。`_lock` 保证每次修改的原子性。

#### 读-判断-写 窗口：存在竞争，但安全 ⚠

```python
# _collect_task_model 中的 TOCTOU 窗口
if not force and state.is_done(task.id, "collect", model_name):  # ① 读
    return                                                        #    ↓ 窗口
collect_single(task, model_name, config, state, ...)             # ② 写
```

两个线程**不会**处理相同的 `(task_id, model_name)` 组合（因为 `submit` 循环为每个组合只创建一个 future），所以这个 TOCTOU 窗口实际上不会被触发。

但如果外部有另一个进程（如另一个终端窗口）同时修改同一个 `pipeline_state.json`，则可能产生冲突——`PipelineState` 不提供跨进程锁。

### 11.6 崩溃安全性

| 崩溃时机 | 后果 | 恢复方式 |
|----------|------|----------|
| batch 期间，`set()` 修改内存后、`save()` 之前 | 所有 batch 内的修改丢失（包括已标记 `"done"` 的条目） | 重跑 `ctpipe collect`，所有任务从头执行；`shutil.copy2` 幂等覆盖，结果正确 |
| `save()` 执行期间（`write_text` 之后、`replace` 之前） | 磁盘上保留旧版本 JSON（`replace` 是原子操作，不会出现半写状态） | 同上 |
| 进程被 `kill -9` | 等同于 batch 期间崩溃 | 同上 |

**设计取舍**：`batch()` 选择了性能而非持久性。batch 期间的修改只存在于内存中，进程崩溃则全部丢失。这在流水线场景下是可接受的——每个阶段都是幂等的（通过 `is_done()` 检查），崩溃后重跑即可。

### 11.7 各阶段 batch 使用模式

| 阶段 | 并发模型 | batch 用法 |
|------|---------|-----------|
| `prepare.py` | 单线程 | `with state.batch():` 包裹所有 task 的 prepare，最后一次性落盘 |
| `run.py` | asyncio 协程 | 不使用 batch（每个 run 结果独立写入，保证持久性） |
| `collect.py` | `ThreadPoolExecutor` | `with state.batch():` 包裹整个线程池执行期间 |
| `score.py` | `asyncio.gather` | `with state.batch():` 包裹所有并发评分协程 |
| `retry.py` | 单线程 | `with state.batch():` 仅包裹 `_reset_entries()` 中的批量 reset |
| `finalize.py` | 单线程 | `with state.batch():` 包裹所有 task 的 finalize |

**`run.py` 不使用 batch 的原因**：每个 run 任务耗时很长（数分钟到数十分钟），如果中途崩溃，丢失已完成的 run 状态代价太高。因此 `run.py` 选择每个任务完成后立即 `save()`，保证持久性。

**`collect.py` 使用 batch 的原因**：collect 操作很快（毫秒级），任务数量可能很多（数百个 task × 2 models = 数百次 `set()`），频繁 `save()` 会产生大量磁盘 I/O。batch 将所有写入合并为一次 `save()`，且 collect 的幂等性使得崩溃后重跑代价很低。

---

## 12 完整状态转换图

```
                     run 阶段
                ┌───────────────────┐
                │   "" → "running"  │
                │        │          │
                │   ┌────┴────┐     │
                │   ▼         ▼     │
                │ "done"   "failed" │
                │ "partial" "error" │
                │          "timeout"│
                └───────┬───────────┘
                        │
           ┌────────────┼────────────────┐
           ▼            ▼                ▼
     collect 正常   collect salvage   collect force
     (run=done)    (run=异常)        (任何 run 状态)
           │            │                │
           ▼            ▼                ▼
    status="done"  status="partial"  run 正常 → "done"
    recovery=F     recovery=T        run 异常 → "partial"
    salvaged=F     salvaged=T        recovery=T, forced=T
    forced=F       forced=F
           │            │                │
           └────────────┼────────────────┘
                        │
                        ▼
              pipeline_state.json
                        │
           ┌────────────┼────────────┐
           ▼            ▼            ▼
        score        finalize      retry
       (阶段 4)     (阶段 5)    (自动重试)
                                     │
                                     ▼
                              级联重置下游
                              重新执行 collect
                              (默认 salvage=True)
                                     │
                              ┌──────┴──────┐
                              ▼             ▼
                          重试成功       超过 max_retries
                        status="done"   status="permanently_failed"
```

---

## 13 CLI 用法汇总

```bash
# 正常 collect（salvage 默认开启）
python -m ctpipe collect

# 禁用 salvage
python -m ctpipe collect --no-salvage

# 强制重收集（绕过所有验证）
python -m ctpipe collect --force

# 针对特定任务和模型
python -m ctpipe collect --tasks CT-0001 CT-0002 --models qwen

# 全流水线执行（collect 使用 salvage=True, force=False）
python -m ctpipe all

# 重置后 force 重收集
python -m ctpipe reset --tasks CT-0001 --stages collect
python -m ctpipe collect --force --tasks CT-0001

# 自动重试（collect 作为级链的一部分被重新执行）
python -m ctpipe retry --max-retries 2
```

---

## 14 测试覆盖

测试文件 `tests/test_collect.py` 包含 16 个单元测试，覆盖正常路径、salvage、force 和错误路径：

| 测试类 | 用例数 | 验证内容 |
|--------|--------|----------|
| `CollectNormalPathTest` | 2 | 正常 collect 成功；run 未完成时跳过 |
| `CollectMissingStartTimeTest` | 2 | start_time 推断（.claude/ mtime）；兜底到 epoch |
| `CollectMissingSessionIdTest` | 2 | session_id 推断（项目哈希目录）；推断失败时继续 |
| `CollectSalvageFromInterruptedRunTest` | 6 | 从 running/failed/timeout salvage；降低行数阈值；session 不匹配降级为 WARN；`--no-salvage` 跳过 |
| `CollectForceRecoveryTest` | 4 | force 绕过 start_time/session_id；force + done 重收集；force 忽略 is_done |
| `CollectRecoveryErrorPathsTest` | 5 | 恢复失败时仍设 recovery 标志；salvage 无 JSONL → skipped；正常失败无 recovery |
