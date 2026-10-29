#!/usr/bin/env python
"""PreToolUse 闸门：在受保护目录被写入前硬拦截。

"反代写"的确定性档位（第 3 档）—— Claude Code 官方文档明确：
CLAUDE.md 只是上下文（建议性），要强制必须用 PreToolUse hook。

三档自 v2 起补齐（此前文档只定义了两级，"中间那档"从未存在）：
  1 建议性    —— CLAUDE.md / SessionStart 注入（learning-mode.sh）
  2 半确定性  —— UserPromptSubmit 每轮提醒（prompt.py）+ Stop 阻断收尾（stop.py）
  3 确定性    —— 本文件

本文件同时是"教学回合"的开关（v2）：
  拦下受保护写入 → 无活动回合则开一个（记 state），要求本人说清打算怎么做；
  答完并落进 09/04、且本人开过口之后 → 放行，回合转入 pending_review；
  之后由 stop.py 守住"检验"（复述 + 四维评分）那一环。
  "本人开过口"由 prompt.py 记录，模型无法伪造 —— 这是防自放行的关键。

覆盖两条通道：
  1. 文件工具（Edit / Write / MultiEdit / NotebookEdit）—— 读 tool_input.file_path
  2. Bash（`>` `>>` 重定向、tee、sed -i、dd of=、cp/mv 目标、touch/rm 等、
     `cd` 进受保护目录后再写、`python -c` 落盘）—— 读 tool_input.command

行为：
  - 命中 → 退出码 2，stderr 回给模型（阻断）
  - 否则 → 退出码 0（放行）

项目根定位（按优先级）：
  1. 命令行第一个参数      ← settings.json 传入绝对路径，最可靠
  2. 环境变量 CLAUDE_PROJECT_DIR
  3. 当前工作目录

  为什么要有 1：官方文档说 hook 进程的 cwd 不保证是项目根，
  而 CLAUDE_PROJECT_DIR 是否注入 hook 进程也未经核实。
  两条都不成立时会找不到清单文件从而静默放行 —— 那是最坏的失败方式。

已知边界（有意保留，避免误伤）：
  命令里不含受保护路径的字面量时看不到，例如 `D=src; echo x > "$D/eval/a.py"`。
  闸门防的是"顺手代写"，不防刻意绕过。

紧急放行：改 .claude/hooks/MODE（on | guard-only | teach-only | off）。
  GUARD-OFF 是弃用别名，仍被尊重，但新写法只有 MODE 一处。
自检：python .claude/hooks/mode.py status
自测：python .claude/hooks/test_guard.py
"""
import json
import os
import re
import sys

# hook 进程跑在"reduced environment"里，不能假设 stdout/stderr 的编码。
# 框架按 UTF-8 解析我们的 JSON 与 stderr，所以显式锁死。
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

# 状态机（教学回合）。导入失败不能连累闸门本身，退回纯拦截行为。
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import _state as state
except Exception:
    state = None

# 写入目标 = 最后一个非选项参数
LAST_ARG_VERBS = {"cp", "mv", "rsync", "install"}
# 写入目标 = 其后所有非选项参数
ALL_ARG_VERBS = {"tee", "touch", "rm", "truncate", "mkdir", "chmod", "chown", "ln"}
# 出现即视为"后续可能写"的动词（配合 cd 使用）
ANY_WRITE_VERBS = LAST_ARG_VERBS | ALL_ARG_VERBS | {"sed", "dd", "python", "python3"}

TOKEN_RE = re.compile(r'"[^"]*"|\'[^\']*\'|\S+')
# 重定向操作符：> >> 1> 2>>   （排除 2>&1 这类）
REDIR_RE = re.compile(r'^\d*>>?$')
# 无空格重定向：把 `>>file` / `1>file` 拆成两个 token。
# 只认 token 开头的 `\d*>>?`，且其后首字符不能是 `&`（否则会把 2>&1 / >&2 拆坏）。
# 残留缺口：`echo x>file` 这种重定向符夹在词中间的写法仍看不到，已记入 test_guard.py。
SPLIT_REDIR_RE = re.compile(r'^(\d*>>?)([^&].*)$')
# heredoc：<<EOF / <<-EOF / <<'PY' / <<"PY"
HEREDOC_RE = re.compile(r'^<<-?[\'"]?([A-Za-z_][A-Za-z0-9_]*)[\'"]?$')
# 会把 heredoc 正文当代码执行的解释器
INTERPRETERS = {
    "python", "python3", "bash", "sh", "zsh",
    "node", "perl", "ruby", "pwsh", "powershell",
}


_MSYS_RE = re.compile(r"^/([A-Za-z])/(.*)$")


def _norm(p: str) -> str:
    r"""Git Bash 的 /d/dev/x → D:/dev/x。

    不归一化会**静默全放行**：Windows 的 Python 把 /d/dev/x 解析成
    <当前盘>:\d\dev\x，于是读不到 protected-paths.txt、mention 为 None、
    直接 return 0 —— 正是本文件开头说的"最坏的失败方式"。
    从 Git Bash 里拿 $PWD 传进来就会踩到（实测过）。

    注意本 docstring 必须是 raw —— 里面的 \d \x 在普通字符串里是非法转义，
    会把整个文件变成 SyntaxError（踩过）。
    """
    s = (p or "").strip().strip('"').strip("'").replace("\\", "/")
    m = _MSYS_RE.match(s)
    return (m.group(1).upper() + ":/" + m.group(2)) if m else s


def resolve_root() -> str:
    if len(sys.argv) > 1 and sys.argv[1].strip():
        return _norm(sys.argv[1])
    env = os.environ.get("CLAUDE_PROJECT_DIR")
    if env and env.strip():
        return _norm(env)
    return _norm(os.getcwd())


def load_protected(hook_dir: str):
    """读取受保护目录列表，返回去掉首尾斜杠的形式。"""
    path = os.path.join(hook_dir, "protected-paths.txt")
    if not os.path.exists(path):
        return []
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            s = s.replace("\\", "/").strip("/")
            if s:
                out.append(s)
    return out


def build_matcher(protected):
    """返回一个函数 s -> bool，按路径段边界匹配，避免 src/evals 与 mysrc/eval 被误判。"""
    if not protected:
        return None
    alts = [r'(?<![\w-])' + re.escape(d) + r'(?![\w-])' for d in protected]
    pat = re.compile("|".join(alts))
    return lambda s: bool(pat.search(s or ""))


def to_rel(path: str, root: str, cwd: str = None):
    """把路径规范化为相对 root 的形式；不在 root 之下返回 None。

    cwd 用于兜住"启动根不等于项目根"的情形：当会话从 plan/ 启动时，
    文件工具可能给出相对 plan/ 的路径（如 MedEvidence/src/eval/a.py），
    只按 root 比会当成普通相对路径漏判。给了 cwd 就先拼成绝对路径再比一次。
    """
    p = (path or "").strip().strip('"').strip("'").replace("\\", "/")
    if not p:
        return None
    while p.startswith("./"):
        p = p[2:]
    r = root.replace("\\", "/").rstrip("/")
    if r and p.lower().startswith(r.lower() + "/"):
        return p[len(r) + 1:]
    if r and p.lower() == r.lower():
        return ""
    if re.match(r'^[A-Za-z]:/', p) or p.startswith("/"):
        return None          # 绝对路径但不在项目内
    if cwd:
        c = cwd.strip().strip('"').strip("'").replace("\\", "/").rstrip("/")
        if c:
            return to_rel(c + "/" + p, root)   # 拼出来是绝对路径，不会再递归
    return p


def is_protected(rel, protected) -> bool:
    if rel is None:
        return False
    for d in protected:
        if rel == d or rel.startswith(d + "/"):
            return True
    return False


def tokenize(cmd: str):
    out = []
    for t in TOKEN_RE.findall(cmd):
        # 先把无空格重定向拆开，否则 `>>src/eval/a.py` 会整体成一个 token 而漏判
        # （引号开头的 token 不拆，避免破坏 "a > b" 这种字面量）
        if t[:1] not in "\"'":
            m = SPLIT_REDIR_RE.match(t)
            if m:
                out.append(m.group(1))
                t = m.group(2)
        if len(t) >= 2 and t[0] == t[-1] and t[0] in "\"'":
            t = t[1:-1]
        out.append(t)
    return out


def split_segments(cmd: str):
    return [s.strip() for s in re.split(r'&&|\|\||[;|\n]', cmd) if s.strip()]


def hits(mention, toks) -> bool:
    """toks 中任一 token 提到受保护路径。"""
    return any(mention(t) for t in toks)


def analyze_segment(cmd: str, toks, mention) -> bool:
    """判断单个命令段是否会写入受保护路径。"""
    if not toks:
        return False
    base = toks[0].replace("\\", "/").rstrip("/").split("/")[-1].lower()

    # 重定向：> target  /  >> target
    for i, t in enumerate(toks):
        if REDIR_RE.match(t) and i + 1 < len(toks):
            if mention(toks[i + 1]):
                return True

    # dd of=...
    if base == "dd":
        for t in toks:
            if t.startswith("of=") and mention(t[3:]):
                return True

    # sed -i（原地改写）
    if base == "sed" and any(t == "-i" or t.startswith("-i") for t in toks):
        if hits(mention, toks[1:]):
            return True

    # cp / mv / rsync：最后一个是目标
    if base in LAST_ARG_VERBS:
        rest = [a for a in toks[1:] if not a.startswith("-")]
        if rest and mention(rest[-1]):
            return True

    # touch / rm / tee / truncate ...
    if base in ALL_ARG_VERBS:
        if hits(mention, toks[1:]):
            return True

    # python -c "..." —— 能力无限，只要提到就算
    if base in ("python", "python3") and any(t == "-c" for t in toks):
        if hits(mention, toks):
            return True

    return False


def bash_blocks(command: str, root: str, protected, mention, cwd: str = None) -> bool:
    if not command or mention is None:
        return False
    # 命令里根本没提到受保护路径 → 一律放行（含拼字符串绕过的情形）
    if not mention(command):
        return False

    cd_into_protected = False
    heredoc_marker = None
    for seg in split_segments(command):
        # heredoc 正文：整段当代码执行，只要提到受保护路径就算
        if heredoc_marker is not None:
            if seg.strip() == heredoc_marker:
                heredoc_marker = None
            elif mention(seg):
                return True
            continue
        toks = tokenize(seg)
        if not toks:
            continue
        # 解释器 + heredoc：正文会走 stdin 被当代码执行，盯住随后几段
        base0 = toks[0].replace("\\", "/").rstrip("/").split("/")[-1].lower()
        if base0 in INTERPRETERS:
            for t in toks:
                m = HEREDOC_RE.match(t)
                if m:
                    heredoc_marker = m.group(1)
                    break
        # cd 进受保护目录
        if toks[0] == "cd" and len(toks) > 1:
            if is_protected(to_rel(toks[1], root, cwd), protected):
                cd_into_protected = True
                continue
        if analyze_segment(seg, toks, mention):
            return True
        # 已经 cd 进受保护目录，后面任何写动词都算
        if cd_into_protected:
            base = toks[0].replace("\\", "/").split("/")[-1].lower()
            if base in ANY_WRITE_VERBS:
                return True
    return False


def emit_block(reason: str, missing=None, root: str = None) -> None:
    """阻断文案。

    `⛔ 这是会被面试追问的目录` 是本文档独有的字符串 —— 文档与自测都靠它
    区分"是闸门拦的"还是"别的东西拦的"，不要改。
    """
    # 只问前瞻问题 —— 代码还没写，回顾性的问题（实测如何、追问三层答得上吗）
    # 归后段的 stop.py。在这里问等于让人编。
    q = state.FRONT_QUESTIONS if state else (
        "动手之前先说清楚（本人说，不要替他答）：\n"
        "  1. 你打算怎么做？为什么这么选？代价是什么？\n"
        "  2. 你打算怎么验证它？（说清指标和实验设计 —— 说方法，不是结果）\n\n")
    r = state.routing(root) if state else (
        "落点：项目同级的 决策记录.md / 怀疑清单.md（旧项目为 09-/04- 前缀；都不存在则跳过落盘）\n\n")
    extra = ""
    if missing:
        extra = ("还差这些条件才会自动放行：\n"
                 + "".join("  ✗ %s\n" % m for m in missing) + "\n")
    sys.stderr.write(
        "⛔ 这是会被面试追问的目录（命中：%s）。\n\n%s%s%s"
        "前瞻两问答完（并按上面的落点要求记录）且本人开过口之后，写入会自动放行。\n"
        "要整体关掉闸门：改 .claude/hooks/MODE（on | guard-only | teach-only | off）。\n"
        % (reason, q, r, extra)
    )


def main() -> int:
    root = resolve_root()
    hook_dir = os.path.join(root, ".claude", "hooks")

    # 单一开关 MODE 优先；_state 不可用时退回旧的 GUARD-OFF 检查
    if state is not None:
        if not state.guard_active():
            return 0
    elif os.path.exists(os.path.join(hook_dir, "GUARD-OFF")):
        return 0

    try:
        payload = json.load(sys.stdin)
    except Exception:
        return 0  # 解析失败一律放行，不阻塞正常开发

    protected = load_protected(hook_dir)
    mention = build_matcher(protected)
    if mention is None:
        return 0

    tool_input = payload.get("tool_input") or {}
    reason = None
    # 会话的当前工作目录（框架注入）。用于兜住"启动根 ≠ 项目根"的相对路径漏判。
    cwd = payload.get("cwd") or ""

    # —— 通道 1：文件工具 ——
    path = (
        tool_input.get("file_path")
        or tool_input.get("notebook_path")
        or tool_input.get("path")
    )
    if path:
        rel = to_rel(path, root, cwd)
        if is_protected(rel, protected):
            reason = rel
    else:
        # —— 通道 2：Bash ——
        command = tool_input.get("command")
        if command and bash_blocks(command, root, protected, mention, cwd):
            reason = "Bash 命令（重定向 / tee / sed -i / cp 目标 / cd 后写入 / heredoc 等）"

    if reason is None:
        return 0

    # —— 教学回合 ——
    # 拦下不只是"拦一次"，它是一个决策点：开一个回合，把产出和检验逼出来。
    # 状态机见 _state.py；判定原则是"闸门守实践入口，stop.py 守检验出口"。
    if state is None:
        emit_block(reason, root=root)     # 状态模块缺失，退回纯拦截
        return 2

    data = state.load()
    rnd = data.get("round")

    if rnd and state.round_is_stale(rnd):
        state.close_round(data, "stale")   # 陈旧回合：清理，避免永久阻断
        rnd = None

    if rnd is None:
        state.open_round(reason, root)
        emit_block(reason, ["本回合刚开始：先说清你打算怎么做，再动手写"], root=root)
        return 2

    if rnd.get("state") == "blocking":
        ok, missing = state.release_ready(rnd, data, root)
        if ok:
            state.release_round(rnd, data, root)
            return 0
        emit_block(reason, missing, root=root)
        return 2

    # pending_review：同一回合内继续写别的受保护文件，直接放行
    return 0


if __name__ == "__main__":
    sys.exit(main())
