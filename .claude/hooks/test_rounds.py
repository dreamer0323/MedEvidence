#!/usr/bin/env python
"""教学回合状态机自测（v2 新增）。

跑法（在项目根目录）：
    python .claude/hooks/test_rounds.py

为什么单独一个文件而不是并进 test_guard.py：这些用例**有跨调用的状态**，
而 test_guard.py 是"一条 payload 一个进程、彼此独立"的纯函数式用例表，
两者混在一起会让那个表变脏。两者都通过 test_guard.py 一个入口跑（见其末尾）。

测试沙箱模拟**真实部署布局**：

    <tmp>/plan/09-决策记录.md           ← 落点（在项目根的上一级）
    <tmp>/plan/04-怀疑清单.md
    <tmp>/plan/Proj/.claude/hooks/*     ← hooks 真的在这里
    <tmp>/plan/Proj/src/eval/           ← 受保护目录

必须复制 hooks 而不是指向原目录：_state.py 把 MODE 与 state/ 写在自己所在目录，
只有 hooks 真的位于 <root>/.claude/hooks/ 时才成立。
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = "src"

HOOK_FILES = ["guard.py", "_state.py", "prompt.py", "stop.py", "stop.sh",
              "learning-mode.sh", "mode.py", "protected-paths.txt"]
DOC_09 = "09-决策记录.md"
DOC_04 = "04-怀疑清单.md"
SID = "test-rounds-session"

fails = []
ran = []


def find_bash():
    """找 Git Bash。

    必须排除 C:\\Windows\\System32\\bash.exe —— 那是 WSL 启动器，在它里面
    跑不了 Windows 的 python。从 Windows Python 起 subprocess 找不到 `bash`
    会命中 System32，本文件踩过这个坑（症状：跳出一段 Microsoft Store /
    "Linux 于 Windows" 的提示）。
    """
    cands = []

    def add(root):
        if root:
            cands.append(os.path.join(root, "bin", "bash.exe"))
            cands.append(os.path.join(root, "usr", "bin", "bash.exe"))

    g = shutil.which("git")
    if g:
        # git 可能位于 <git>/cmd/、<git>/bin/ 或 <git>/mingw64/bin/，
        # 所以一路往上找，而不是只看两层
        d = os.path.dirname(g)
        for _ in range(4):
            add(d)
            parent = os.path.dirname(d)
            if parent == d:
                break
            d = parent

    for p in (r"C:\Program Files\Git", r"C:\Program Files (x86)\Git",
              os.path.expandvars(r"%LOCALAPPDATA%\Programs\Git")):
        add(p)

    for c in cands:
        if c and os.path.isfile(c) and "system32" not in c.lower():
            return c
    return None


BASH = find_bash()


def check(name, cond, extra=""):
    ran.append(name)
    print("[%s] %s%s" % ("OK  " if cond else "FAIL", name,
                         "" if cond else "   <-- " + str(extra)))
    if not cond:
        fails.append(name)


def _first_protected(proj):
    """取沙箱里 protected-paths.txt 的第一个受保护目录（相对项目根）。

    不能写死 src/eval —— 清单是按项目定的，写死会让换项目后测试全红。
    """
    p = os.path.join(proj, ".claude", "hooks", "protected-paths.txt")
    try:
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                s = line.strip()
                if s and not s.startswith("#"):
                    return s.rstrip("/")
    except OSError:
        pass
    return "%s/eval" % SRC            # 兜底：模板原始清单的第一项


def build_sandbox():
    tmp = tempfile.mkdtemp(prefix="rounds-")
    plan = os.path.join(tmp, "plan")
    proj = os.path.join(plan, "Proj")
    hooks = os.path.join(proj, ".claude", "hooks")
    os.makedirs(hooks)
    for f in HOOK_FILES:
        s = os.path.join(HERE, f)
        if os.path.exists(s):
            shutil.copy2(s, os.path.join(hooks, f))
    first = _first_protected(proj)
    os.makedirs(os.path.join(proj, first))
    for name in (DOC_09, DOC_04):
        with open(os.path.join(plan, name), "w", encoding="utf-8") as fh:
            fh.write("# %s\n\n初始内容\n" % name)
    return tmp, plan, proj, hooks, first


def env_for(proj):
    e = dict(os.environ)
    e["CLAUDE_PROJECT_DIR"] = proj
    e["CLAUDE_CODE_SESSION_ID"] = SID
    return e


def call(proj, script, payload=None, argv_extra=None):
    """跑一个 hook。.py 用 python，.sh 用（显式找到的）Git Bash。"""
    path = os.path.join(proj, ".claude", "hooks", script)
    if script.endswith(".sh"):
        cmd = [BASH or "bash", path]
    else:
        cmd = [sys.executable, path] + (argv_extra or [])
    return subprocess.run(cmd, input=json.dumps(payload or {}),
                          capture_output=True, text=True,
                          encoding="utf-8", errors="replace", env=env_for(proj))


def guard(proj, file_path):
    return call(proj, "guard.py", {"cwd": proj, "tool_input": {"file_path": file_path}},
                argv_extra=[proj])


# 有 Git Bash 时连哨兵一起测；没有就退回直接测 stop.py 的逻辑
STOP_SCRIPT = "stop.sh" if BASH else "stop.py"


def stop(proj, script=None):
    """跑 Stop hook，返回解析后的 JSON（解析不了就返回 {}）。"""
    r = call(proj, script or STOP_SCRIPT,
             {"session_id": SID, "stop_hook_active": False, "cwd": proj})
    try:
        return json.loads(r.stdout or "")
    except Exception:
        return {}


def state_of(proj):
    p = os.path.join(proj, ".claude", "hooks", "state", "session-%s.json" % SID)
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def write_state(proj, data):
    p = os.path.join(proj, ".claude", "hooks", "state", "session-%s.json" % SID)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)


def append(plan, name, text):
    with open(os.path.join(plan, name), "a", encoding="utf-8") as fh:
        fh.write(text)


def set_mode(proj, value):
    with open(os.path.join(proj, ".claude", "hooks", "MODE"), "w", encoding="utf-8") as fh:
        fh.write(value + "\n")


def clear_mode(proj):
    try:
        os.unlink(os.path.join(proj, ".claude", "hooks", "MODE"))
    except OSError:
        pass


def main():
    tmp, plan, proj, hooks, first = build_sandbox()
    target = first + "/chunk.py"

    try:
        # 1. 首次写受保护目录 → 拦，且开回合、带前瞻两问
        r = guard(proj, target)
        check("1 首次写入被拦 (rc=2)", r.returncode == 2, "rc=%s" % r.returncode)
        check("1 前段只问前瞻问题",
              "你打算怎么做" in r.stderr and "你打算怎么验证" in r.stderr, r.stderr[:200])
        # 回归：曾经在写入前就问"怎么验证它是对的""追问三层答得上哪层" ——
        # 那两问是回顾性的，代码还没写，只能编。这是被用户实测骂过的具体错误。
        check("1 前段不问回顾性问题",
              "我怎么验证它是对的" not in r.stderr
              and "我答得上哪一层" not in r.stderr, r.stderr[:300])
        check("1 阻断文案含独有标记", "⛔ 这是会被面试追问的目录" in r.stderr)
        d = state_of(proj)
        check("1 回合已开启且为 blocking",
              (d.get("round") or {}).get("state") == "blocking", d.get("round"))
        check("1 哨兵 open-* 已建",
              os.path.exists(os.path.join(hooks, "state", "open-%s" % SID)))

        # 2. 只回「继续」、什么都不做 → 仍被拦
        r = guard(proj, target)
        check("2 只回「继续」仍被拦", r.returncode == 2)
        check("2 报出缺「记录」", "还没有本回合的新增记录" in r.stderr, r.stderr[-200:])

        # 3. 我（模型）自己往 09 写一条，但本人没开口 → 仍被拦 ← 防自放行
        append(plan, DOC_09, "\n### 99. 测试决策\n- **决策**：x\n")
        r = guard(proj, target)
        check("3 只落盘、本人没开口 → 仍被拦", r.returncode == 2)
        check("3 报出缺「本人开口」", "本人还没开口" in r.stderr, r.stderr[-200:])

        # 4. 本人开口（UserPromptSubmit）→ 两条齐备 → 放行
        call(proj, "prompt.py", {"prompt": "我为什么选这个做法？因为…", "cwd": proj})
        r = guard(proj, target)
        check("4 落盘 + 本人开口 → 放行 (rc=0)", r.returncode == 0, r.stderr[:200])
        d = state_of(proj)
        check("4 回合转入 pending_review",
              (d.get("round") or {}).get("state") == "pending_review", d.get("round"))

        # 5. 想收尾 → Stop 拦住（检验没做）
        out = stop(proj)
        check("5 Stop 阻断收尾", out.get("decision") == "block", out)
        reason5 = out.get("reason") or ""
        check("5 后段问回顾性问题", "实测结果" in reason5, reason5[:200])
        check("5 后段不问前瞻问题（分工不能反）", "你打算怎么做" not in reason5, reason5[:200])
        check("5 后段按面试相关性分档", "不看笔记" in reason5 and "低相关" in reason5,
              reason5[:300])
        check("5 阻断计数已记", state_of(proj)["counters"].get("stop_blocks") == 1)

        # 6. 本人再次开口（=复述发生了）→ Stop 放行并关回合
        call(proj, "prompt.py", {"prompt": "它解决的是…它怎么工作…", "cwd": proj})
        out = stop(proj)
        check("6 复述后 Stop 放行", not out, out)
        d = state_of(proj)
        check("6 回合已关闭", d.get("round") is None, d.get("round"))
        check("6 哨兵已清除",
              not os.path.exists(os.path.join(hooks, "state", "open-%s" % SID)))

        # 6b. 哨兵路径：没有开放回合时，stop.sh 必须静默早退
        #     （stderr 也不能有噪音 —— 曾经因为 `< MODE` 的重定向报错漏到 stderr）
        if BASH:
            r = call(proj, "stop.sh", {"session_id": SID, "cwd": proj})
            check("6b 无回合时 stop.sh 静默早退",
                  r.returncode == 0 and r.stdout.strip() == "" and r.stderr.strip() == "",
                  "rc=%s out=%r err=%r" % (r.returncode, r.stdout[:80], r.stderr[:120]))

        # 7. 安全阀：Stop 最多拦 2 次
        guard(proj, target)
        append(plan, DOC_09, "\n### 100. 测试决策\n- **决策**：y\n")
        call(proj, "prompt.py", {"prompt": "答", "cwd": proj})
        guard(proj, target)                     # 放行 → pending_review
        outs = [stop(proj) for _ in range(3)]
        n_block = sum(1 for o in outs if o.get("decision") == "block")
        check("7 最多拦 2 次后自动放弃", n_block == 2, "实际 %d 次" % n_block)
        check("7 触顶已记账", state_of(proj)["counters"].get("stop_block_limits") == 1)
        check("7 触顶后回合已关", state_of(proj).get("round") is None)

        # 8. 安全阀：陈旧回合（超 TTL）被清理重开，不会永久阻断
        guard(proj, target)
        d = state_of(proj)
        d["round"]["opened_at"] = "2020-01-01T00:00:00"      # 远超 TTL
        write_state(proj, d)
        r = guard(proj, target)
        check("8 陈旧回合被清理并重开", r.returncode == 2 and
              state_of(proj)["round"]["opened_at"] != "2020-01-01T00:00:00")

        # 9. 单一开关：off → 全放行
        set_mode(proj, "off")
        r = guard(proj, target)
        check("9 MODE=off → 闸门放行", r.returncode == 0, r.stderr[:120])

        # 10. guard-only → 闸门还在，但不要求"本人开口"（因为每轮提醒已关）
        set_mode(proj, "guard-only")
        r = guard(proj, target)
        check("10 MODE=guard-only → 仍拦", r.returncode == 2)
        check("10 且不再要求本人开口", "本人还没开口" not in r.stderr, r.stderr[-200:])
        r = call(proj, "prompt.py", {"prompt": "x", "cwd": proj})
        check("10 guard-only 下 prompt.py 不注入", r.stdout.strip() == "", r.stdout[:120])

        # 11. 四档行为矩阵
        # 曾经的真实 bug：learning-mode.sh 只 gate 了"记账"、没 gate"输出"，
        # 导致 MODE=off / guard-only 下学习模式照样注入 —— 四个档坏了两个。
        matrix = {
            "on":         {"gate": True,  "prompt": True,  "inject": True},
            "guard-only": {"gate": True,  "prompt": False, "inject": False},
            "teach-only": {"gate": False, "prompt": True,  "inject": True},
            "off":        {"gate": False, "prompt": False, "inject": False},
        }
        for mode in ("on", "guard-only", "teach-only", "off"):
            want = matrix[mode]
            set_mode(proj, mode)
            gate = guard(proj, target).returncode == 2
            pr = call(proj, "prompt.py", {"prompt": "x", "cwd": proj}).stdout.strip() != ""
            inj = call(proj, "learning-mode.sh",
                       {"source": "startup", "cwd": proj, "session_id": SID}).stdout.strip() != ""
            check("11 MODE=%-10s 闸门=%s 提醒=%s 注入=%s"
                  % (mode, "拦" if gate else "放行", "有" if pr else "无", "有" if inj else "无"),
                  gate == want["gate"] and pr == want["prompt"] and inj == want["inject"],
                  "期望 gate=%s prompt=%s inject=%s" % (want["gate"], want["prompt"], want["inject"]))
            clear_mode(proj)

    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print()
    if fails:
        print("%d/%d 通过，%d 项失败：" % (len(ran) - len(fails), len(ran), len(fails)))
        for n in fails:
            print("  - %s" % n)
        return 1
    print("%d/%d 通过（教学回合状态机 + 四档矩阵）" % (len(ran), len(ran)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
