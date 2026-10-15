"""MedEvidence：证据性安全性特化的医学文献 RAG Agent。

子包职责（见 plan/08-技术方案.md 第六节）：

    config          集中配置（唯一读 os.environ 的地方）
    embedding       BGE-M3 向量化封装
    vectorstore     Milvus Lite 封装
    ingest          抓取 + XML 解析 + 结构化分块   ← 受保护
    retrieve        检索 + 重排                    ← 受保护
    generate        Prompt 组装 + LLM 调用         ← 受保护（周 3）
    eval            评测指标                       ← 受保护（周 4）
    api             FastAPI 服务                   ← 周 8
"""
