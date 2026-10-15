#!/usr/bin/env python
"""PreToolUse 闸门：在受保护目录被写入前硬拦截。

这是"反代写"三档里唯一的确定性档位 —— Claude Code 官方文档明确：
CLAUDE.md 只是上下文（建议性），要强制必须用 PreToolUse hook。

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

紧急放行：在项目里创建 .claude/hooks/GUARD-OFF 即可关闭本闸门。
自测：python .claude/hooks/test_guard.py
"""
import json
import os
import re
import sys

# 写入目标 = 最后一个非选项参数
LAST_ARG_VERBS = {"cp", "mv", "rsync", "install"}
# 写入目标 = 其后所有非选项参数
ALL_ARG_VERBS = {"tee", "touch", "rm", "truncate", "mkdir", "chmod", "chown", "ln"}
# 出现即视为"后续可能写"的动词（配合 cd 使用）
ANY_WRITE_VERBS = LAST_ARG_VERBS | ALL_ARG_VERBS | {"sed", "dd", "python", "python3"}

TOKEN_RE = re.compile(r'"[^"]*"|\'[^\']*\'|\S+')
# 重定向操作符：> >> 1> 2>>   （排除 2>&1 这类）
REDIR_RE = re.compile(r'^\d*>>?$')


def resolve_root() -> str:
    if len(sys.argv) > 1 and sys.argv[1].strip():
        return sys.argv[1].strip()
    env = os.environ.get("CLAUDE_PROJECT_DIR")
    if env and env.strip():
        return env.strip()
    return os.getcwd()


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


def to_rel(path: str, root: str):
    """把路径规范化为相对 root 的形式；不在 root 之下返回 None。"""
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


def bash_blocks(command: str, root: str, protected, mention) -> bool:
    if not command or mention is None:
        return False
    # 命令里根本没提到受保护路径 → 一律放行（含拼字符串绕过的情形）
    if not mention(command):
        return False

    cd_into_protected = False
    for seg in split_segments(command):
        toks = tokenize(seg)
        if not toks:
            continue
        # cd 进受保护目录
        if toks[0] == "cd" and len(toks) > 1:
            if is_protected(to_rel(toks[1], root), protected):
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


def main() -> int:
    root = resolve_root()
    hook_dir = os.path.join(root, ".claude", "hooks")

    if os.path.exists(os.path.join(hook_dir, "GUARD-OFF")):
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

    # —— 通道 1：文件工具 ——
    path = (
        tool_input.get("file_path")
        or tool_input.get("notebook_path")
        or tool_input.get("path")
    )
    if path:
        rel = to_rel(path, root)
        if is_protected(rel, protected):
            reason = rel
    else:
        # —— 通道 2：Bash ——
        command = tool_input.get("command")
        if command and bash_blocks(command, root, protected, mention):
            reason = "Bash 命令（重定向 / tee / sed -i / cp 目标 / cd 后写入 等）"

    if reason is None:
        return 0

    sys.stderr.write(
        "⛔ 这是会被面试追问的目录（命中：%s）。\n\n"
        "在动手之前，先用三句话回答：\n"
        "  1. 我为什么选这个做法？代价是什么？\n"
        "  2. 我怎么验证它是对的？（指标 / 实验）\n"
        "  3. 面试官追问三层时，我答得上哪一层、答不上哪一层？\n\n"
        "答完把答案写进 plan/09-决策记录.md，再让本人回复「继续」，本次即可放行。\n"
        "若确实需要长期放行，由本人手动创建空文件 .claude/hooks/GUARD-OFF。\n" % reason
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
