#!/usr/bin/env python
"""guard.py 自测：跑完所有用例并打印通过/失败。

用法（在项目根目录）：
    python .claude/hooks/test_guard.py
退出码非 0 表示有用例失败。
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
GUARD = os.path.join(HERE, "guard.py")
SRC = "src"   # 拆开拼，避免本文件被自己的闸门拦下
PLAN = os.path.dirname(ROOT)              # 上一层，模拟"从 plan/ 启动会话"
PROJ = os.path.basename(ROOT)             # 项目目录名，用于拼相对 plan 根的路径

# 自测的隔离目录，由 main() 建、main() 删。
# 不隔离会污染生产状态：guard.py 会通过 _state 记账，而 CLAUDE_CODE_SESSION_ID
# 从父 shell 继承，等于把一条假回合写进你正在用的会话里。
_STATE_TMP = None
_TEST_SID = "test-guard"

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
    ("文件: embedding 受保护 → 拦截",
     {"tool_input": {"file_path": f"{SRC}/embedding/client.py"}}, True),
    ("文件: vectorstore 受保护 → 拦截",
     {"tool_input": {"file_path": f"{SRC}/vectorstore/milvus_store.py"}}, True),
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

    # --- v2 新增：无空格重定向 ---
    ("Bash: 无空格 >> 重定向 → 拦截",
     {"tool_input": {"command": f"echo x >>{SRC}/eval/a.py"}}, True),
    ("Bash: 无空格 1> 重定向 → 拦截",
     {"tool_input": {"command": f"echo x 1>{SRC}/eval/a.py"}}, True),
    ("Bash: 无空格 >> 但 2>&1 未被拆坏（对照）",
     {"tool_input": {"command": f"echo hi 2>&1 > {SRC}/eval/a.py"}}, True),
    ("Bash: 已知缺口2(重定向符夹在词中) → 放行",
     {"tool_input": {"command": f"echo x>{SRC}/eval/a.py"}}, False),

    # --- v2 新增：heredoc 正文 ---
    ("Bash: python heredoc 落盘 → 拦截",
     {"tool_input": {"command": f"python - <<'PY'\nopen('{SRC}/eval/x.py','w').write('')\nPY"}}, True),
    ("Bash: heredoc 不碰受保护路径 → 放行",
     {"tool_input": {"command": "python - <<'PY'\nprint(1)\nPY"}}, False),

    # --- v2 新增：启动根 ≠ 项目根（相对 plan/ 的路径）---
    ("路径: 相对 plan 根 → 拦截",
     {"cwd": PLAN,
      "tool_input": {"file_path": f"{PROJ}/{SRC}/eval/a.py"}}, True),
    ("路径: 相对 plan 根但不涉受保护 → 放行",
     {"cwd": PLAN,
      "tool_input": {"file_path": f"{PROJ}/scratch/ok.py"}}, False),
]


def run(payload):
    env = dict(os.environ, CLAUDE_PROJECT_DIR=ROOT)
    env["CLAUDE_CODE_SESSION_ID"] = _TEST_SID        # 不蹭真实会话
    if _STATE_TMP:
        env["HOOKS_STATE_DIR"] = _STATE_TMP          # 状态写到临时目录
    proc = subprocess.run(
        [sys.executable, GUARD],
        input=json.dumps(payload), capture_output=True, text=True,
        encoding="utf-8", errors="replace", env=env,
    )
    return proc.returncode


def main() -> int:
    global _STATE_TMP
    _STATE_TMP = tempfile.mkdtemp(prefix="guardtest-")
    failed = 0
    try:
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
        print(f"\n{len(CASES) - failed}/{len(CASES)} 通过（闸门纯函数式用例）")

        # MSYS 路径回归：从 Git Bash 里拿 $PWD（形如 /d/dev/x）传进来时，
        # Windows 的 Python 解析不了 → 读不到 protected-paths.txt → **静默全放行**。
        # 这正是 guard.py 开头说的"最坏的失败方式"，实测踩到过。
        msys = "/" + ROOT[0].lower() + ROOT[2:].replace("\\", "/")
        env = dict(os.environ, CLAUDE_PROJECT_DIR=ROOT, CLAUDE_CODE_SESSION_ID=_TEST_SID,
                   HOOKS_STATE_DIR=_STATE_TMP)
        proc = subprocess.run(
            [sys.executable, GUARD, msys],
            input=json.dumps({"cwd": ROOT, "tool_input": {"file_path": f"{SRC}/eval/a.py"}}),
            capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)
        ok = proc.returncode == 2
        print(f"[{'OK  ' if ok else 'FAIL'}] {'MSYS 路径的 root 不能静默放行':38s} "
              f"rc={proc.returncode}  arg={msys}")
        if not ok:
            failed += 1
    finally:
        shutil.rmtree(_STATE_TMP, ignore_errors=True)
        _STATE_TMP = None

    # 教学回合状态机的用例在 test_rounds.py —— 那些有跨调用的状态，
    # 与上面这张"一条 payload 一个进程、彼此独立"的表混在一起会把它弄脏。
    # 这里代为调用一次，保证 `python .claude/hooks/test_guard.py` 仍是唯一入口。
    rounds = os.path.join(HERE, "test_rounds.py")
    if os.path.exists(rounds):
        print("\n--- 教学回合状态机（test_rounds.py）---")
        proc = subprocess.run([sys.executable, rounds], text=True,
                              encoding="utf-8", errors="replace")
        if proc.returncode != 0:
            failed += 1

    print()
    print("=" * 46)
    print("闸门自测" + ("全部通过" if failed == 0 else "有失败项"))
    print("=" * 46)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
