"""数据清单：记录"这批语料 + 这份评测集"的状态。

用法（项目根目录）：
    python -m src.eval.manifest

它回答的问题是：**这个数字，是在哪份数据、哪份代码上跑出来的？**

为什么不用 git 就够：`.gitignore` 把 `data/*` 排除了（语料 XML、Milvus 库、
评测集都不在 git 里），所以 git 只记录代码，记录不了数据状态。
manifest 补上数据这一半 —— 两者结合才叫"可复现"。
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path

from src.config import DATA_DIR, EMBEDDING_DIM, EVAL_DATASET, RAW_DIR
from src.vectorstore import MilvusStore

MANIFEST_PATH = DATA_DIR / "manifest.json"


def git_commit() -> str:
    """当前 commit；有未提交改动时加 `-dirty`。拿不到就返回 "unknown"。"""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()

        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()

        if dirty:
            return commit + "-dirty"
        return commit

    except Exception:
        return "unknown"


def file_sha256(path: Path) -> str:
    """文件内容的 sha256 —— 数据集改一个字，这个值就变。"""
    digest = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            block = f.read(1 << 20)
            if not block:
                break
            digest.update(block)

    return digest.hexdigest()


def build_manifest(store: MilvusStore) -> dict:
    """生成清单内容（不写盘）。"""
    n_papers = len(list(RAW_DIR.glob("*.xml")))

    return {
        "built_at": datetime.now().isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "n_papers": n_papers,
        "n_chunks": store.count(),
        "embedding_dim": EMBEDDING_DIM,
        "dataset_sha256": file_sha256(EVAL_DATASET),
    }


def read_manifest() -> dict:
    """读已有清单；没有就返回空 dict（run.py 用它填 corpus_chunks）。"""
    if not MANIFEST_PATH.exists():
        return {}
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def main() -> int:
    store = MilvusStore()
    store.ensure_collection()
    manifest = build_manifest(store)
    store.close()

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"\n已写入 {MANIFEST_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
