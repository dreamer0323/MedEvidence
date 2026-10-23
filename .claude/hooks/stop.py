#!/usr/bin/env python
"""Stop：回合没关就不许收尾。

由 stop.sh 在确认"存在开放回合"之后调用，所以这里可以放心地做重活。

拦的是后段：代码已经放行写入了（pending_review），但检验还没做。
闸门守在"实践"的入口，这里守在"检验"的出口。

安全阀（防把人困住）：
  · 必查 stop_hook_active（框架自带，防死循环）
  · 每回合最多拦 MAX_STOP_BLOCKS 次，超出则自动放弃并记账
  · 陈旧回合（超 TTL）直接清理
  · MODE=guard-only / off 时 stop.sh 根本不会走到这里
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


def main() -> int:
    if state is None or not state.teach_active():
        return 0
    try:
        payload = json.load(sys.stdin)
    except Exception:
        return 0

    # 框架自带：由 Stop hook 自己触发的那一轮不要再拦，否则死循环
    if payload.get("stop_hook_active"):
        return 0

    data = state.load()
    rnd = data.get("round")
    if not rnd:
        state.clear_open()
        return 0
    if state.round_is_stale(rnd):
        state.close_round(data, "stale")
        return 0
    if rnd.get("state") != "pending_review":
        return 0                      # 还在等本人说清打算怎么做，收尾是合理的

    # 检验完成的两种可判信号：落点又变了，或本人放行后开过口
    if state.review_done(rnd) or state.spoke_after_release(data, rnd):
        state.close_round(data, "reviewed")
        return 0

    if rnd.get("stop_blocks", 0) >= state.MAX_STOP_BLOCKS:
        data["counters"]["stop_block_limits"] = data["counters"].get("stop_block_limits", 0) + 1
        state.close_round(data, "stop_block_limit")
        return 0

    rnd["stop_blocks"] = rnd.get("stop_blocks", 0) + 1
    data["counters"]["stop_blocks"] = data["counters"].get("stop_blocks", 0) + 1
    state.save(data)
    left = state.MAX_STOP_BLOCKS - rnd["stop_blocks"]

    print(json.dumps({
        "decision": "block",
        "reason": (
            "回合还没关：受保护目录（%s）的代码已经放行写入，现在该收尾了。\n\n"
            "%s%s"
            "（本回合还剩 %d 次阻断，之后会自动放行并记账。）"
            % (rnd.get("target", "受保护目录"),
               state.BACK_QUESTIONS, state.BACK_DEPTH, left)
        ),
        "systemMessage": "⛔ 教学回合未完成：代码已放行，还差收尾评估。",
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
