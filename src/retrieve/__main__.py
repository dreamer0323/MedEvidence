"""人眼探针：python -m src.retrieve "你的问题" [top_k]

只打印，不生成答案 —— 用来判断"这 5 条相关吗"。
第 7 周 badcase 归因就从这里开始：不相关是分块的问题还是检索的问题。
"""
import sys

from src.retrieve import retrieve


def main() -> int:
    if len(sys.argv) < 2:
        print('用法：python -m src.retrieve "你的问题" [top_k]')
        return 1

    query = sys.argv[1]
    top_k = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    hits = retrieve(query, top_k=top_k)
    if not hits:
        print("没有命中：库是空的？还是 query 为空？")
        return 0

    for i, hit in enumerate(hits, 1):
        print(f"[{i}] {hit.score:.4f}  {hit.cite()} / {hit.path}")
        print("    " + hit.text[:120].replace("\n", " "))
    return 0


if __name__ == "__main__":
    sys.exit(main())
