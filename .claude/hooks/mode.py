#!/usr/bin/env python
"""单一开关 + 确定性自检。

用法（在项目根目录）：
    python .claude/hooks/mode.py status          自检报告（只读磁盘证据，不问模型）
    python .claude/hooks/mode.py on              全套：注入 + 每轮提醒 + 闸门 + 收尾强制
    python .claude/hooks/mode.py guard-only      只留闸门；放行只看记录是否落盘
    python .claude/hooks/mode.py teach-only      只注入教学文本，不拦
    python .claude/hooks/mode.py off             全关

为什么要它：原来"启用/关闭"分散在四处（GUARD-OFF、删 SessionStart 段、删
PreToolUse 段、删 .claude/），粒度还各不相同。现在所有 hook 统一读 .claude/hooks/MODE
这一个文件。GUARD-OFF 作为弃用别名保留，仍会被尊重。
"""
import json
import os
import sys

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import _state as state
except Exception as exc:          # pragma: no cover
    sys.stderr.write("无法导入 _state.py：%s\n" % exc)
    sys.exit(1)


def _fmt_round(rnd) -> str:
    if not rnd:
        return "无（idle）"
    label = {"blocking": "拦截中（等本人说清打算怎么做）",
             "pending_review": "待检验（代码已放行，还差复述评分）"}
    return "%s · 目标 %s · 开启于 %s · 已阻断收尾 %d 次" % (
        label.get(rnd.get("state"), rnd.get("state") or "?"),
        rnd.get("target", "?"),
        rnd.get("opened_at", "?"),
        rnd.get("stop_blocks", 0),
    )


def status() -> int:
    mode = state.get_mode()
    off = state.guard_off_present()
    root = state.project_root()
    parent = os.path.dirname(os.path.abspath(root.rstrip("/\\")))

    print("=" * 62)
    print("开关")
    print("=" * 62)
    LABEL = {
        "on": "全套：注入 + 每轮提醒 + 闸门 + 收尾强制",
        "guard-only": "只留闸门：不注入、不提醒、不收尾（放行只看记录是否落盘）",
        "teach-only": "只留教学：注入 + 提醒；不拦、不收尾",
        "off": "全关",
    }
    print("  MODE                : %s  —— %s%s" % (
        mode, LABEL[mode],
        "   （文件缺失，按缺省 on）" if not os.path.exists(state.MODE_FILE) else ""))
    print("  GUARD-OFF（弃用别名）: %s" % (
        "存在 —— 它只关闸门；与 MODE 冲突时以「关」为准。建议删掉它，统一用 MODE"
        if off else "不存在"))
    print("  实际生效            : 闸门 %s · 教学注入 %s" % (
        "开" if state.guard_active() else "关",
        "开" if state.teach_active() else "关"))
    print("  切换                : python .claude/hooks/mode.py <on|guard-only|teach-only|off>")
    print("                        （立即生效 —— 这些 hook 每次调用都重读 MODE，不用重开会话）")

    print()
    print("=" * 62)
    print("项目根 / 启动根")
    print("=" * 62)
    print("  项目根（hook 视角）      : %s" % root)
    print("  落点目录                 : %s" % parent)
    for p in state.doc_paths(root):
        print("    %-22s %s" % (os.path.basename(p), "存在" if os.path.exists(p) else "**不存在**"))
    print("  protected-paths         :")
    try:
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "protected-paths.txt"), encoding="utf-8") as fh:
            for line in fh:
                s = line.strip()
                if s and not s.startswith("#"):
                    print("    %s" % s)
    except OSError:
        print("    （读不到 protected-paths.txt）")

    print()
    print("=" * 62)
    print("会话证据（来自 state/，不是问模型）")
    print("=" * 62)
    paths = state.all_sessions()
    if not paths:
        print("  state/ 里没有任何会话记录。")
        print("  → 说明 SessionStart 从未在这个项目里跑过，学习模式注入大概率没生效。")
    for i, p in enumerate(paths[:5]):
        try:
            with open(p, encoding="utf-8") as fh:
                d = json.load(fh)
        except Exception:
            print("  （%s 读不出来）" % os.path.basename(p))
            continue
        tag = "最新" if i == 0 else "第 %d 新" % (i + 1)
        print("  [%s] %s" % (tag, os.path.basename(p)))
        cwd = d.get("cwd_at_start") or "（未记录）"
        print("      启动根      : %s" % cwd)
        if cwd not in ("（未记录）", ""):
            same = os.path.abspath(cwd).lower() == os.path.abspath(root).lower()
            print("      与项目根一致: %s%s" % (
                "是" if same else "**否**",
                "" if same else "   ← 嵌套子目录的 .claude/settings.json 不会被搜到，hook 可能没加载"))
        inj = d.get("injections") or []
        if inj:
            for it in inj[-4:]:
                print("      注入        : %s  source=%-8s %s 字符" % (
                    it.get("at", "?"), it.get("source", "?"), it.get("chars", "?")))
            srcs = {it.get("source") for it in inj}
            if "compact" in srcs:
                print("      → 见过 source=compact：压缩上下文后重新注入过，模式没丢")
        else:
            print("      注入        : **无记录**（SessionStart 没跑成，或 MODE=off/guard-only）")
        print("      用户输入    : %d 条（最近 %s）" % (
            d.get("user_turn_count") or 0, d.get("last_user_turn") or "-"))
        print("      当前回合    : %s" % _fmt_round(d.get("round")))
        c = d.get("counters") or {}
        print("      记账        : 拦截 %s · 放行 %s · 阻断收尾 %s · 触顶放弃 %s" % (
            c.get("blocks", 0), c.get("grants", 0),
            c.get("stop_blocks", 0), c.get("stop_block_limits", 0)))
        hist = d.get("history") or []
        if hist:
            print("      最近回合史  : %s" % ", ".join(
                "%s(%s)" % (h.get("closed_reason", "?"), h.get("target", "?")) for h in hist[-3:]))
    stales = [p for p in paths if _stale(p)]
    if stales:
        print()
        print("  陈旧会话文件（超过 %d 小时的回合已失效，可删）：" % state.TTL_HOURS)
        for p in stales[:5]:
            print("    %s" % os.path.basename(p))

    print()
    print("=" * 62)
    print("判定要点")
    print("=" * 62)
    print("  · 拦截必须能看到 `⛔ 这是会被面试追问的目录` —— 那是 guard.py 独有的字符串")
    print("  · 改了 .claude/ 下的 settings.json 要重开会话；改 guard.py 等脚本立即生效")
    return 0


def _stale(p: str) -> bool:
    try:
        with open(p, encoding="utf-8") as fh:
            d = json.load(fh)
    except Exception:
        return False
    rnd = d.get("round")
    if not rnd:
        return False
    return state.round_is_stale(rnd)


def main() -> int:
    args = sys.argv[1:]
    if not args or args[0] == "status":
        return status()
    if args[0] == "set":
        if len(args) < 2:
            sys.stderr.write("用法: mode.py set <%s>\n" % "|".join(state.MODE_VALUES))
            return 1
        state.set_mode(args[1])
        print("MODE = %s" % state.get_mode())
        return 0
    if args[0] in state.MODE_VALUES:
        state.set_mode(args[0])
        print("MODE = %s" % state.get_mode())
        return 0
    sys.stderr.write(
        "用法:\n"
        "  mode.py status\n"
        "  mode.py <%s>\n" % "|".join(state.MODE_VALUES))
    return 1


if __name__ == "__main__":
    sys.exit(main())
