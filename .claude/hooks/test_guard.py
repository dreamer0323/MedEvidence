#!/usr/bin/env python
"""guard.py 自测：跑完所有用例并打印通过/失败。

用法（在项目根目录）：
    python .claude/hooks/test_guard.py
退出码非 0 表示有用例失败。

**用例的目录是从 `.claude/hooks/protected-paths.txt` 动态取的**，
不绑定任何具体项目——换项目、改清单后本测试依然有意义。
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


def _load_protected():
    """从 protected-paths.txt 读受保护目录（跳过空行与注释）。"""
    p = os.path.join(HERE, "protected-paths.txt")
    out = []
    try:
        with open(p, encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                out.append(s.rstrip("/"))
    except OSError:
        pass
    return out


PROTECTED = _load_protected()


def D(i):
    """取第 i 个受保护目录；不够就复用最后一个；清单为空返回 None。"""
    if not PROTECTED:
        return None
    return PROTECTED[min(i, len(PROTECTED) - 1)]


def build_cases():
    """(说明, hook payload, 期望是否拦截)。清单为空时只保留"应放行"用例。"""
    cases = []
    first, second = D(0), D(1)

    # --- 文件工具通道：应拦 / 不应误伤 ---
    if first:
        win_first = first.replace("/", chr(92))
        top, _, rest = first.partition("/")
        rest = rest or "probe"
        cases += [
            ("文件: 绝对路径命中", {"tool_input": {"file_path": f"{ROOT}/{first}/probe.py"}}, True),
            ("文件: 相对路径命中", {"tool_input": {"file_path": f"{first}/probe.py"}}, True),
            ("文件: Windows 反斜杠命中", {"tool_input": {"file_path": f"{ROOT}\\{win_first}\\probe.py"}}, True),
            ("文件: 目录本身命中", {"tool_input": {"file_path": f"{first}"}}, True),
            ("文件: 相似目录（尾加 s）→ 不误伤", {"tool_input": {"file_path": f"{first}s/probe.py"}}, False),
            ("文件: 前缀不同（my+路径）→ 不误伤", {"tool_input": {"file_path": f"my{first}/probe.py"}}, False),
            ("文件: 项目外同名目录 → 不误伤", {"tool_input": {"file_path": f"D:/other/my{SRC}/{rest}/x.py"}}, False),
            ("文件: 笔记里提到该路径 → 不误伤",
             {"tool_input": {"file_path": f"{ROOT}/notes/关于{first}/的思考.md"}}, False),
            ("文件: scratch 放行", {"tool_input": {"file_path": f"{ROOT}/scratch/page_demo.py"}}, False),
        ]
        if second and second != first:
            cases.append(("文件: 清单里第二个目录也拦",
                          {"tool_input": {"file_path": f"{second}/probe.py"}}, True))

        # --- Bash 通道：应拦 ---
        target2 = second or first
        cases += [
            ("Bash: 重定向写文件", {"tool_input": {"command": f"cat > {first}/probe.py <<EOF"}}, True),
            ("Bash: 追加重定向", {"tool_input": {"command": f"echo x >> {target2}/a.py"}}, True),
            ("Bash: sed -i 原地改", {"tool_input": {"command": f"sed -i 's/a/b/' {first}/probe.py"}}, True),
            ("Bash: tee", {"tool_input": {"command": f"tee {first}/probe.py"}}, True),
            ("Bash: cd 后 touch", {"tool_input": {"command": f"cd {first} && touch x.py"}}, True),
            ("Bash: cp 目标在受保护目录", {"tool_input": {"command": f"cp /tmp/b.py {first}/a.py"}}, True),
            ("Bash: dd of=", {"tool_input": {"command": f"dd if=/dev/zero of={first}/big.bin"}}, True),
            ("Bash: python -c 落盘",
             {"tool_input": {"command": f"python -c \"open('{first}/x.py','w')\""}}, True),
            ("Bash: 绝对路径写入 → 拦截",
             {"tool_input": {"command": f"echo x > {ROOT}/{first}/a.py"}}, True),
            ("Bash: 无空格 >> 重定向 → 拦截", {"tool_input": {"command": f"echo x >>{first}/a.py"}}, True),
            ("Bash: 无空格 1> 重定向 → 拦截", {"tool_input": {"command": f"echo x 1>{first}/a.py"}}, True),
            ("Bash: 无空格 >> 但 2>&1 未被拆坏（对照）",
             {"tool_input": {"command": f"echo hi 2>&1 > {first}/a.py"}}, True),
            ("Bash: python heredoc 落盘 → 拦截",
             {"tool_input": {"command": f"python - <<'PY'\nopen('{first}/x.py','w').write('')\nPY"}}, True),
        ]

        # --- Bash 通道：应放行 ---
        cases += [
            ("Bash: ls 只读", {"tool_input": {"command": f"ls {first}/"}}, False),
            ("Bash: grep 只读", {"tool_input": {"command": f"grep -r rerank {first}/"}}, False),
            ("Bash: cp 源在受保护目录（读）", {"tool_input": {"command": f"cp {first}/a.py /tmp/b.py"}}, False),
            ("Bash: 不涉及受保护路径", {"tool_input": {"command": "ls -la && git status"}}, False),
            ("Bash: 2>&1 不是写入", {"tool_input": {"command": "echo hi 2>&1 | tee /tmp/x.log"}}, False),
            ("Bash: heredoc 不碰受保护路径", {"tool_input": {"command": "python - <<'PY'\nprint(1)\nPY"}}, False),
            ("Bash: 已知缺口（路径被拼出）→ 放行",
             {"tool_input": {"command": f'D={top}; echo x > "$D/{rest}/a.py"'}}, False),
            ("Bash: 已知缺口2（重定向符夹在词中）→ 放行",
             {"tool_input": {"command": f"echo x>{first}/a.py"}}, False),
        ]

        # --- 启动根 ≠ 项目根 ---
        cases += [
            ("路径: 相对 plan 根 → 拦截",
             {"cwd": PLAN, "tool_input": {"file_path": f"{PROJ}/{first}/a.py"}}, True),
            ("路径: 相对 plan 根但不涉受保护 → 放行",
             {"cwd": PLAN, "tool_input": {"file_path": f"{PROJ}/scratch/ok.py"}}, False),
        ]

        # --- 动态覆盖：清单里每个目录都要能被拦 ---
        for d in PROTECTED:
            cases.append((f"动态: {d} 必须被拦",
                          {"tool_input": {"file_path": f"{d}/probe.py"}}, True))
    else:
        cases += [
            ("清单为空：不误伤普通文件", {"tool_input": {"file_path": f"{ROOT}/{SRC}/x.py"}}, False),
            ("清单为空：不误伤 Bash", {"tool_input": {"command": f"echo x > {SRC}/x.py"}}, False),
        ]
    return cases


CASES = build_cases()


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
        src = "、".join(PROTECTED) if PROTECTED else "（空）"
        print(f"受保护目录（来自 protected-paths.txt）：{src}\n")
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
        if PROTECTED:
            msys = "/" + ROOT[0].lower() + ROOT[2:].replace("\\", "/")
            env = dict(os.environ, CLAUDE_PROJECT_DIR=ROOT, CLAUDE_CODE_SESSION_ID=_TEST_SID,
                       HOOKS_STATE_DIR=_STATE_TMP)
            proc = subprocess.run(
                [sys.executable, GUARD, msys],
                input=json.dumps({"cwd": ROOT, "tool_input": {"file_path": f"{PROTECTED[0]}/a.py"}}),
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
