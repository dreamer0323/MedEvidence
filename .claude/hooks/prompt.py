#!/usr/bin/env python
"""UserPromptSubmit：每轮注入短提醒 + 记录用户输入时间戳。

两个作用：
  1. 抗上下文漂移 —— 把"每场会话注入一次"改成"每轮都在场"。
     第 1 条抱怨（软约束不稳定）最便宜的一刀。
  2. 为 guard 的放行条件提供证据 —— 用户输入是模型无法伪造的信号，
     所以"我替你写好记录再自己放行"这条路被堵死。

成本：每轮一次进程。相比挂在 Edit|Write 上的 PostToolUse（每回合 5-20 次），
这个乘数可以接受。
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
except Exception:
    state = None


def build(data) -> str:
    rnd = (data or {}).get("round")
    if rnd and not state.round_is_stale(rnd):
        if rnd.get("state") == "blocking":
            return (
                "⛔ 有未完成的回合：闸门正拦着 `%s`。\n"
                "先让本人说清两件**前瞻**的事（本人说，不要替他答）："
                "① 打算怎么做、为什么、代价是什么；② 打算怎么验证（说方法，不是结果）。"
                "答案落进 09/04 且本人开过口之后，写入就会自动放行。不要跳过。"
                % rnd.get("target", "受保护目录")
            )
        if rnd.get("state") == "pending_review":
            return (
                "⛔ 有未完成的收尾：受保护目录的代码已放行写入。\n"
                "现在问**回顾**的：① 实测结果如何、和原计划差在哪；"
                "② 面试官追问这个选择的三层，答得上哪几层。\n"
                "再按**面试相关性**定深度 —— 高相关才走不看笔记复述四问 + 四维评分，"
                "低相关一句话说清就过。定级理由要落进 04/09。"
            )
    return (
        "（无活动教学回合）会被面试追问的部分让本人写 5-10 行有意义代码，不要代写；\n"
        "唯一准则是高效拿到入职 —— 低相关的活快速放行，别把每个改动都做成教案。"
    )


def main() -> int:
    if state is None or not state.teach_active():
        return 0
    try:
        payload = json.load(sys.stdin)
    except Exception:
        return 0

    try:
        data = state.load()
        data["user_turn_count"] = (data.get("user_turn_count") or 0) + 1
        data["last_user_turn"] = state.iso(state.now())
        state.save(data)
        ctx = build(data)
    except Exception:
        # 记账失败不能连累提醒；退回无状态文案
        ctx = ("（无活动教学回合）会被面试追问的部分让本人写 5-10 行有意义代码，不要代写。")

    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": ctx,
        }
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
