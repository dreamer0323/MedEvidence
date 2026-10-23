# 数据模型（一页纸）

> **一句话**：XML 进来 → `Section` → `Chunk` → `ChunkRow` → Milvus。
> 前三个是内存里的临时对象，**只有最后一步落盘**。
> 冲突时以代码为准；改结构先改这里。

---

## 0 · 30 秒版

1. **四个结构，一个落盘**：`Article/Section`、`Chunk`、`ChunkRow` 都是内存对象，
   持久化的只有 Milvus 里的行。
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
