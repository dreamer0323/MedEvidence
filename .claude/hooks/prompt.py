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


LEARN_PRE = (
    "⚡ 本人本条输入命中「学习模式」触发词，**学习模式的技能正文已随上下文提供**。\n"
    "以下是本回合的硬性格式，违反即本回合失败：\n"
    "1. 你的**第一条回复第一行**必须是：\n"
    "   `项目：<名> ｜ 仓库：<路径> ｜ 计划根：<路径> ｜ 入口：<A 仓库根 / B 计划根>`\n"
    "   自报之前不得输出任何其它内容。\n"
    "2. 按 learn 技能 §0 判定 cwd 属于 A/B/C，并用工具调用验证；判定失败或文件缺失 →\n"
    "   只输出 `⛔ 路径校验失败：…` 两行然后停止。\n"
    "3. **不要**声称 `/learn` 未安装或未执行；**不要**用「你想学什么 / 要动哪块」反问——\n"
    "   主题按技能 §2 由台账与任务卡推导。\n"
)


# ---------------------------------------------------------------- 路径解析
# 由 hook 自己算，不让模型推：模型只需原样打印 + 用一次工具复核。

def _has_repo_marker(p: str) -> bool:
    return os.path.exists(os.path.join(p, "AGENTS.md")) or os.path.exists(os.path.join(p, ".git"))


def resolve(cwd: str) -> dict:
    """按 learn 技能 §0 的 A/B/C 规则解析，返回可用/不可用与各项路径。"""
    out = {"cwd": cwd, "case": "C", "repo": None, "plan": None, "ws": None,
           "name": None, "ok": False, "missing": []}
    if not cwd or not os.path.isdir(cwd):
        out["missing"].append("cwd 不存在")
        return out

    if _has_repo_marker(cwd):
        out.update(case="A", repo=cwd, plan=os.path.dirname(cwd.rstrip("\\/")))
    elif os.path.isdir(os.path.join(cwd, "learn")):
        try:
            projs = [d for d in sorted(os.listdir(os.path.join(cwd, "learn")))
                     if os.path.isdir(os.path.join(cwd, "learn", d))]
        except OSError:
            projs = []
        if len(projs) == 1:
            x = projs[0]
            out.update(case="B", plan=cwd)
            for cand in (os.path.join(cwd, "code"),
                         os.path.join(cwd, x),
                         os.path.join(cwd, x, "code")):
                if os.path.isdir(cand) and _has_repo_marker(cand):
                    out["repo"] = cand
                    break
        elif len(projs) > 1:
            out["missing"].append("learn\\ 下有多个项目：%s" % "、".join(projs))
            return out
        else:
            out["missing"].append("learn\\ 下没有项目")
            return out
    else:
        out["missing"].append("既不是仓库根（无 AGENTS.md/.git）也不是计划根（无 learn\\）")
        return out

    if not out["repo"]:
        out["missing"].append("B 类下找不到仓库目录（code / <项目> / <项目>\\code 都不含 AGENTS.md 或 .git）")
        return out

    leaf = os.path.basename(out["repo"].rstrip("\\/"))
    out["name"] = os.path.basename(os.path.dirname(out["repo"].rstrip("\\/"))) if leaf == "code" else leaf
    out["ws"] = os.path.join(out["plan"], "learn", out["name"])

    for f in ("MISSION.md", "NOTES.md", "RESOURCES.md"):
        if not os.path.exists(os.path.join(out["ws"], f)):
            out["missing"].append("%s\\%s" % (out["name"], f))
    for f in ("决策记录.md", "09-决策记录.md", "怀疑清单.md", "04-怀疑清单.md"):
        pass  # 台账二选一，单独判定
    ledgers = [os.path.join(out["plan"], f) for f in
               ("决策记录.md", "09-决策记录.md", "怀疑清单.md", "04-怀疑清单.md")]
    if not any(os.path.exists(p) for p in ledgers):
        out["missing"].append("计划根下没有决策记录/怀疑清单")

    out["ok"] = not out["missing"]
    return out


def learn_block(cwd: str) -> str:
    """给模型的硬注入：hook 已经算好，模型只负责打印与复核。"""
    r = resolve(cwd)
    if r["ok"]:
        line = ("项目：%s ｜ 仓库：%s ｜ 计划根：%s ｜ 入口：%s"
                % (r["name"], r["repo"], r["plan"],
                   "A 仓库根" if r["case"] == "A" else "B 计划根"))
        return (
            "✅ 路径已由 hook 解析完成（**不要自己再推**，也不要改成别的路径）：\n"
            "```\n%s\n```\n"
            "要求：\n"
            "- 回复的**第一行必须原样打印上面这一行**（一个字都不要改）。\n"
            "- 然后**先用一次工具调用**复核（例如 `ls \"%s\"` 看 MISSION.md 在不在）；\n"
            "  复核与上面一致 → 按 learn 技能 §1 加载状态并开课；不一致 → 输出 ⛔ 两行并停止。\n"
            "- **紧接着第二段必须是今日主题，不许跳过**：\n"
            "  ```\n  今日主题：<一句话说清今天学什么>\n  理由：<指向台账/任务卡的具体证据，例如「学习进度里 rerank 只拿了 7 分」>\n  ```\n"
            "  写完这两条**再**进入诊断或讲解。**禁止**用「你想学什么」「要动哪块」反问。\n"
            "- 入口是 **B** 时：先 `cd \"%s\"`（或用绝对路径），再谈任何代码。\n"
            % (line, r["ws"], r["repo"])
        )
    return (
        "⛔ hook 的路径校验**失败**。请让回复的**第一行**就是：\n"
        "```\n⛔ 路径校验失败：%s。当前 cwd=%s（判定为 %s）。\n```\n"
        "然后停止，不要开课、不要列举选项、不要用历史内容替代。\n"
        % ("；".join(r["missing"]) or "未知原因", r["cwd"], r["case"])
    )



def is_learn_trigger(text: str) -> bool:
    """本人显式要求进入学习模式。命中才注入格式硬要求，避免日常对话被干扰。"""
    t = (text or "").strip()
    if not t:
        return False
    for p in ("/learn", "开始学习", "进入学习", "学习模式"):
        if t.startswith(p):
            return True
    return False


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

    # 触发词要在读完 stdin 后判定；payload 里带本人这条输入的原文
    raw = ""
    if isinstance(payload, dict):
        raw = payload.get("prompt") or payload.get("user_prompt") or ""
    triggered = is_learn_trigger(raw)

    try:
        data = state.load()
        data["user_turn_count"] = (data.get("user_turn_count") or 0) + 1
        data["last_user_turn"] = state.iso(state.now())
        state.save(data)
        ctx = build(data)
    except Exception:
        # 记账失败不能连累提醒；退回无状态文案
        ctx = ("（无活动教学回合）会被面试追问的部分让本人写 5-10 行有意义代码，不要代写。")

    if triggered:
        try:
            ctx = learn_block(os.getcwd()) + "\n" + ctx
        except Exception:
            ctx = LEARN_PRE + "\n" + ctx

    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": ctx,
        }
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
