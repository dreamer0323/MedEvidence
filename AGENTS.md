# MedEvidence

> 本文件是**跨 harness 的单一真源**。Claude Code 通过同目录的 `CLAUDE.md`（内容只有一行
> `AGENTS.md`）指到这里；Codex 直接读本文件。**改这里，两边同时生效。**

## 一、项目目标

**产出**：一个针对**证据性安全性**特化的医学文献 RAG Agent —— 数据源为 PMC Open Access
全文，答案必须**带引用**，并配套一套自建评测集（Recall@5 / MRR / 幻觉率 / 失败归因）。

**给谁看**：2027 届校招 AI 应用开发 / Agent 开发岗的面试官。
所以本仓库有两个等价产出：**一个能跑的系统**，和**一条能被追问的技术判断链**
（判断链落在 `../09-决策记录.md`，面试话术直接从那里长出来）。

**技术骨架**（定死，见 `../08-技术方案.md`）：
结构化分块 → BGE-M3 向量 + BM25 混合检索 → RRF 融合 → bge-reranker 重排
→ LangGraph 多 Agent 编排（Supervisor + 术语/检索/核查）→ DeepSeek 生成 → FastAPI 服务。

## 二、学习模式（反代写）

本项目已开启学习模式，由 `.claude/hooks/learning-mode.sh` 在每次会话启动时注入。
核心分工：

| 区域 | AI 怎么用 |
|---|---|
| **会被面试追问的部分** | 让本人写 5–10 行有意义代码，或先让本人给方案 |
| **不会被追问的支撑工作** | AI 直接做，不必征求同意 |

受保护目录清单见 `.claude/hooks/protected-paths.txt`。
写入受保护目录前，`guard.py` 会硬拦截，要求先回答三问；答完并落进 `09`/`04` 之后自动放行。

**开关只有一处**：`.claude/hooks/MODE`（`on` / `guard-only` / `teach-only` / `off`），
用 `python .claude/hooks/mode.py status` 查看、`python .claude/hooks/mode.py <值>` 切换。
改完**立即生效**，不用重开会话。自检报告里带注入证据和当前回合状态。

闸门覆盖**两条通道**：文件工具（`Edit/Write/MultiEdit/NotebookEdit`）和
Bash（`>` 重定向、`tee`、`sed -i`、`cp/mv` 目标、`dd of=`、`python -c`/heredoc）。
自测命令：`python .claude/hooks/test_guard.py`（27 个用例，改动闸门后必跑）。

> ⚠️ **已知缺口**（有意保留，避免误伤）：命令里不含受保护路径的**字面量**时看不到，
> 例如 `D=src; echo x > "$D/eval/a.py"`。闸门防的是"顺手代写"，不防刻意绕过。

## 三、接口约定（AI 猜不到、必须写下来的）

> 数据在各层之间长什么样、谁拥有、落不落盘：见 `docs/data-model.md`（一页纸）。

- 环境：Windows 11，Git Bash；Python 用 `python`（不是 `python3`）
- 虚拟环境：`.venv`
- 密钥：全部走 `.env`，**任何情况下不得写进代码或提交**
- 数据：只用公开数据，**不碰公司数据**

### 常用命令

```bash
# 虚拟环境（Windows Git Bash；用 `python`，不是 `python3`）
source .venv/Scripts/activate

# 装依赖
pip install -r requirements.txt

# 密钥：复制模板后填真实值（.env 已被 gitignore，永不提交）
cp .env.example .env
```

> ⚠️ `pip install "pymilvus[milvus_lite]"` 在 Windows 上**不生效**，必须单独
> `pip install milvus-lite`，否则 `MilvusClient('x.db')` 报 `ConnectionConfigException`。

```bash
# 待填（src/ 建好后再补）：
#   python -m src.ingest.build      # 离线建库
#   python -m src.eval.run          # 跑评测集
#   uvicorn src.api.main:app --reload   # 起服务
```

## 四、进度与决策

- 决策记录（决策 / 理由 / 代价 / 来源）：`../09-决策记录.md`
- 未解疑问：`../04-怀疑清单.md`
- **每次做完一个技术选型，必须先写进决策记录，再继续。**
