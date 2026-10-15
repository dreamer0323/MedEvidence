# MedEvidence：循证医疗文献检索问答系统

一项基于 RAG 技术的医疗文献类 agent，针对**证据性安全性**特化的智能体系统 ——
数据源为 PMC Open Access 全文，**答案必须带引用**，并配套自建评测集。

> 技术方案、周任务卡、决策记录都在上一级 `plan/` 目录：
> `plan/08-技术方案.md`（用什么）、`plan/02-周任务卡.md`（什么时候做）、
> `plan/09-决策记录.md`（为什么这么选 —— **面试话术从这里长出来**）。

## 目录结构

```
MedEvidence/
├── src/
│   ├── config.py         集中配置（唯一读 os.environ 的地方）
│   ├── embedding/        BGE-M3 向量化封装
│   ├── vectorstore/      Milvus Lite 封装
│   ├── ingest/           抓取 + XML 解析 + 结构化分块    ← 第 2 周
│   ├── retrieve/         检索 + 重排                     ← 第 2 周起
│   ├── generate/         Prompt 组装 + LLM 调用          ← 第 3 周
│   ├── eval/             评测指标                        ← 第 4 周
│   └── api/              FastAPI 服务                    ← 第 8 周
├── data/                 原始 XML / 分块结果 / Milvus Lite（**不进 git**）
├── tests/                单元测试
├── scratch/              临时脚本、试验田
└── conftest.py           pytest 的项目根锚点（勿删）
```

> `data/` 被 gitignore 整体忽略，clone 下来不会有这些目录 —— 跑一次
> `python -m src.config` 就会重建 `data/raw`、`data/processed`、`log/`。

## 快速开始

```bash
# 1. 虚拟环境（Windows Git Bash；用 python，不是 python3）
python -m venv .venv
source .venv/Scripts/activate

# 2. 依赖
pip install -r requirements.txt
pip install -r requirements-dev.txt   # 只有 pytest；⚠️ 直连 pypi 不通时加 -i 镜像

# 3. 密钥（.env 已被 gitignore，永不提交）
cp .env.example .env                  # 然后填真实值
```

> ⚠️ 本机 pip 直连 `pypi.org` 会 SSL 中断，装包加清华镜像：
> `pip install -i https://pypi.tuna.tsinghua.edu.cn/simple <包名>`

## 自检命令（不花钱、不联网）

```bash
python -m src.config                        # 环境变量是否齐全（密钥只打尾巴）
python -W ignore -m src.vectorstore.milvus_store   # 向量库能否连上、建集合
python -m pytest -q                         # 16 个单测
```

需要联网的一次性确认（会花掉几次 API 调用）：

```bash
python -W ignore -m src.embedding.client    # 真调一次 BGE-M3，确认维度 = 1024
```

## 第 2 周进度（数据管线：抓取 → 分块 → 入库 → 能检索）

- [x] 项目骨架：配置层 / embedding 封装 / Milvus Lite 封装 / 单测
- [ ] `src/ingest/`：esearch + efetch 抓 PMC OA 全文（1000 篇）
- [ ] `src/ingest/`：XML 章节解析 + **结构化分块**（分块参数需自行决定）
- [ ] `src/ingest/build.py`：离线建库入口
- [ ] `src/retrieve/`：向量检索跑通，能 print 出条数 + 检索出相关段落
