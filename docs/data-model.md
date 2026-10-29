# 数据模型（一页纸）

> **一句话**：XML 进来 → `Section` → `Chunk` → `ChunkRow` → Milvus。
> 前三个是内存里的临时对象，**只有最后一步落盘**。
> 冲突时以代码为准；改结构先改这里。

---

## 0 · 30 秒版

1. **四个结构，一个落盘**：`Article/Section`、`Chunk`、`ChunkRow` 都是内存对象，
   持久化的只有 Milvus 里的行。
   在线半段（`Hit` / `CitationReport`）见第 7 节，同样**不落盘**。
2. **每层拥有自己的类**：`Section` ≠ `Chunk` ≠ `ChunkRow`；转换只在 `build.to_row()`。
3. **路径只有一个来源**：`src/config.py`；其他模块不自己拼。

---

## 1 · 生命周期

| 步骤 | 结构 | 在哪 | 谁产生 | 落盘 |
|---|---|---|---|---|
| 0 | `data/raw/*.xml` | 磁盘 | esearch / efetch | ✅ |
| 1 | `Article` / `Section` | 内存 | `parse_pmc.parse_article` | ❌ |
| 2 | `Chunk` | 内存 | `chunk.build_chunks` | ❌ |
| 3 | `ChunkRow` | 内存 | `build.to_row` | ❌ |
| 4 | Milvus collection | `data/medevidence.db/` | `MilvusStore.insert` | ✅ |

```
XML ──lxml+清洗──▶ Article/Section ──分块──▶ Chunk ──to_row+embed──▶ ChunkRow ──insert──▶ Milvus
       parse_pmc                    chunk.py                    build.py                  milvus_store.py
```

---

## 2 · 字段对照：`Chunk` → `ChunkRow`

| `Chunk` | `ChunkRow` | 转换规则 |
|---|---|---|
| `text`（纯正文） | `text` | 原样；≤ 8192 字符 |
| `section`（末级标题） | `section` | 原样；≤ 256 字符 |
| `path`（列表） | `path`（字符串） | `" / ".join(path)` |
| `pmc_id` / `chunk_index` | 同名 | 原样 |
| `source_type` | `source_type` | `sec` / `abstract` |
| `embed_input`（property） | **不存** | 仅用于喂 embedding 和算 hash |
| —— | `vector` | `embed(chunk.embed_input)` |
| —— | `text_hash` | `sha256(chunk.embed_input)` |

---

## 3 · 落盘清单

| 路径 | 内容 | 说明 |
|---|---|---|
| `data/raw/*.xml` | 原始语料 | 只读，不修改 |
| `data/medevidence.db/` | Milvus Lite | **目录**，不是单个文件 |
| `log/` | 运行日志 | 按日期滚动 |

派生数据（`Article` / `Section` / `Chunk`）**不落盘**：每次从 XML 重建。
否则就是"第二份真相"——改一次解析代码，还要同步一次快照。

---

## 4 · 踩坑表（都是真踩过的）

| # | 坑 | ❌ 错误写法 | ✅ 正确写法 |
|---|---|---|---|
| 1 | `Section.path` **已含末级标题** | `path + [section]` → `Methods / X / X` | 只拼 `path` |
| 2 | `embed_input` 是 `@property` | `chunk.embed_input()` → TypeError | `chunk.embed_input` |
| 3 | hash 的对象 | `sha256(chunk.text)` | `sha256(chunk.embed_input)` |
| 4 | `path` 的存储类型 | 传 `list` 进 VARCHAR 列 | `" / ".join(path)` |
| 5 | 章节层级 | 只存末级标题就丢了层级 | 末级进 `section`，全链进 `path` |

---

## 5 · 路径的唯一来源

- 所有路径都在 `src/config.py` 定义，其他模块只 import、不自己拼。
- `MILVUS_URI` 走环境变量：云端与本地可以指向不同路径。
- Milvus Lite 是**嵌入式**的：**建库进程必须和 `.db` 在同一台机器**，
  不能远程连；跨机器只能同步文件。

---

## 6 · 增量建库的两条策略

| 策略 | 怎么做 | 适用 | 代价 |
|---|---|---|---|
| **A（现在）** | 新论文增量；改清洗/分块/解析 → `--drop` 全量 | 语料冻结 + 代码迭代 | 代码一改就全量重算 |
| **B（升级）** | 按 `pmc_id` 对账：新 hash 插入、消失的 hash 按 `id` 删除 | 需要长期增量 | 多一个 `rows_by_pmc()` 和删除逻辑 |

`existing_hashes()` 只够 A；B 还要能拿到库里行的 `id`。

---

## 7 · 在线半段：Milvus → 答案（第 3 周新增）

> **一句话**：一次检索的结果是 `Hit`；它的编号 `[n]` 只活在一次 `generate()` 里，不落盘。

### 7.1 生命周期

| 步骤 | 结构 | 谁产生 | 落盘 | 生命周期 |
|---|---|---|---|---|
| 1 | `ChunkRow`（Milvus 行） | `build.to_row` | ✅ | 永久 |
| 2 | `Hit` | `retrieve.retrieve` | ❌ | 单次检索 |
| 3 | 编号 `[n]` | `build_prompt` | ❌ | 单次 `generate()` |
| 4 | `answer` 字符串 | DeepSeek | ❌ | 单次请求 |
| 5 | `CitationReport` | `verify_citations` | ❌ | 单次答案 |

```
Milvus 行 (ChunkRow)
  pk · vector · text · pmc_id · section · chunk_index · path · source_type · text_hash
        │
        │ store.search(query_vector, top_k)
        ▼
Hit  (text · pmc_id · section · chunk_index · path · source_type · score)
        │
        │ build_prompt(question, evidence)      ← 在这里建立 [1..n]
        ▼
Prompt 字符串
        │
        │ DeepSeek（temperature=0）
        ▼
answer 字符串（含 [n]）
        │
        │ verify_citations(answer, 同一份 evidence)
        ▼
CitationReport(used · invalid · unsupported · uncited)
```

### 7.2 字段对照：`ChunkRow` → `Hit`

| ChunkRow | Hit | 说明 |
|---|---|---|
| `text` | `text` | 原样进 prompt |
| `pmc_id` | `pmc_id` | 溯源 |
| `section` | `section` | 末级标题 |
| `chunk_index` | `chunk_index` | 篇内第几块 |
| `path` | `path` | 全链路径 |
| `source_type` | `source_type` | `sec` / `abstract` |
| `vector` | **不传** | 只在库内算相似度 |
| `text_hash` | **不传** | 只用于建库去重 |
| `pk` | **不传** | 建库行身份，在线链路不用 |
| —— | `score` | 检索时才算，不落盘 |

### 7.3 三个最容易混的点

1. **`pk` ≠ `Hit`** —— `pk` 是"库里哪一行"（`{pmc_id}:{chunk_index}`），
   `Hit` 是"这次检索到的证据"。`pk` 不进 prompt。
2. **`[n]` 不是 ID，是位置** —— 由 `build_prompt` 按 `evidence` 的**当前顺序**编号，
   `verify_citations` 用**同一份列表**解释。中间过滤 / 重排 → 静默错位。
3. **`CitationReport` 不落盘** —— 它是一次答案的体检结果；第 4 周评测时才被收集成数据集。

### 7.4 改一处会串到哪

| 改动 | 影响 |
|---|---|
| `retrieve` 的 `top_k` | 只变 `Hit` 数量，不影响编号规则 |
| 中间加 `min_score` 过滤 | ⚠️ 必须在 `retrieve` 内部或 `build_prompt` 之前做完，否则编号错位 |
| 改 `CHUNK_HARD_LIMIT` | `text` 变长 → prompt 变大 → 成本上升；需 `--drop` 重建 |
| 改 `SYSTEM_PROMPT` | 只影响生成，不影响检索与校验 |
