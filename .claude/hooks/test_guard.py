#!/usr/bin/env python
"""guard.py 自测：跑完所有用例并打印通过/失败。

用法（在项目根目录）：
    python .claude/hooks/test_guard.py
退出码非 0 表示有用例失败。
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
GUARD = os.path.join(HERE, "guard.py")
SRC = "src"   # 拆开拼，避免本文件被自己的闸门拦下

# (说明, hook payload, 期望是否拦截)
CASES = [
    # --- 文件工具通道 ---
    ("文件: 绝对路径命中",
     {"tool_input": {"file_path": f"{ROOT}/{SRC}/retrieve/hybrid.py"}}, True),
    ("文件: 相对路径命中",
     {"tool_input": {"file_path": f"{SRC}/retrieve/hybrid.py"}}, True),
    ("文件: Windows 反斜杠命中",
     {"tool_input": {"file_path": f"{ROOT}\\{SRC}\\eval\\run.py"}}, True),
    ("文件: 目录本身命中",
     {"tool_input": {"file_path": f"{SRC}/eval"}}, True),
    ("文件: agent 目录命中",
     {"tool_input": {"file_path": f"{SRC}/agent/graph.py"}}, True),
    ("文件: 笔记里提到该路径 → 不误伤",
     {"tool_input": {"file_path": f"{ROOT}/notes/关于{SRC}/eval/的思考.md"}}, False),
    ("文件: 项目外同名目录 → 不误伤",
     {"tool_input": {"file_path": "D:/other/my" + SRC + "/eval/x.py"}}, False),
    ("文件: embedding 未受保护 → 放行",
     {"tool_input": {"file_path": f"{SRC}/embedding/bge.py"}}, False),
    ("文件: scratch 放行",
     {"tool_input": {"file_path": f"{ROOT}/scratch/page_demo.py"}}, False),

    # --- Bash 通道：应拦 ---
    ("Bash: 重定向写文件",
     {"tool_input": {"command": f"cat > {SRC}/eval/run.py <<EOF"}}, True),
    ("Bash: 追加重定向",
     {"tool_input": {"command": f"echo x >> {SRC}/retrieve/a.py"}}, True),
    ("Bash: sed -i 原地改",
     {"tool_input": {"command": f"sed -i 's/a/b/' {SRC}/retrieve/hybrid.py"}}, True),
    ("Bash: tee",
     {"tool_input": {"command": f"tee {SRC}/ingest/chunk.py"}}, True),
    ("Bash: cd 后 touch",
     {"tool_input": {"command": f"cd {SRC}/eval && touch x.py"}}, True),
    ("Bash: cp 目标在受保护目录",
     {"tool_input": {"command": f"cp /tmp/b.py {SRC}/retrieve/a.py"}}, True),
    ("Bash: dd of=",
     {"tool_input": {"command": f"dd if=/dev/zero of={SRC}/ingest/big.bin"}}, True),
    ("Bash: python -c 落盘",
     {"tool_input": {"command": f"python -c \"open('{SRC}/eval/x.py','w')\""}}, True),

    # --- Bash 通道：应放行 ---
    ("Bash: ls 只读",
     {"tool_input": {"command": f"ls {SRC}/retrieve/"}}, False),
    ("Bash: grep 只读",
     {"tool_input": {"command": f"grep -r rerank {SRC}/eval/"}}, False),
    ("Bash: python -m 跑模块",
     {"tool_input": {"command": f"python -m {SRC}.eval.run"}}, False),
    ("Bash: cp 源在受保护目录",
     {"tool_input": {"command": f"cp {SRC}/retrieve/a.py /tmp/b.py"}}, False),
    ("Bash: 2>&1 不是写入",
     {"tool_input": {"command": f"echo hi 2>&1 | tee /tmp/x.log"}}, False),
    ("Bash: 不涉及受保护路径",
     {"tool_input": {"command": "ls -la && git status"}}, False),

    # --- Bash 通道：路径段边界（防误伤 / 防漏判）---
    ("Bash: 相似目录 src/evals → 放行",
     {"tool_input": {"command": f"cd {SRC}/evals && touch x.py"}}, False),
    ("Bash: 相似目录 mysrc/eval → 放行",
     {"tool_input": {"command": f"cd my{SRC}/eval && touch x.py"}}, False),
    ("Bash: 绝对路径写入 → 拦截",
     {"tool_input": {"command": f"echo x > {ROOT}/{SRC}/eval/a.py"}}, True),
    ("Bash: 已知缺口(路径被拼出) → 放行",
     {"tool_input": {"command": f'D={SRC}; echo x > "$D/eval/a.py"'}}, False),
]


def run(payload):
    env = dict(os.environ, CLAUDE_PROJECT_DIR=ROOT)
    proc = subprocess.run(
        [sys.executable, GUARD],
        input=json.dumps(payload), capture_output=True, text=True, env=env,
    )
    return proc.returncode


def main() -> int:
    failed = 0
    for name, payload, want_block in CASES:
        rc = run(payload)
        blocked = rc == 2
        ok = blocked == want_block
        if not ok:
            failed += 1
        mark = "OK  " if ok else "FAIL"
        got = "拦截" if blocked else "放行"
        exp = "拦截" if want_block else "放行"
        note = "" if rc in (0, 2) else f"  <-- 闸门异常退出 rc={rc}"
        print(f"[{mark}] {name:38s} rc={rc} got={got} want={exp}{note}")
    print(f"\n{len(CASES) - failed}/{len(CASES)} 通过")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
