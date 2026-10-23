#!/usr/bin/env bash
# 学习模式：项目级 SessionStart hook，向上下文注入"反代写"行为约束。
#
# 机制与文本结构改编自 Anthropic 官方插件 learning-output-style
# (github.com/anthropics/claude-plugins-official, Apache License 2.0)。
# 原文注入的是通用划分；本文件把划分替换为用户自己的"会被追问 / 不会被追问"，
# 并补上官方内置 Learning output style 里被漏掉的几条硬规则。许可证全文见同目录 NOTICE。
#
# 同时记录注入证据（时间 / source / 字符数 / 启动根），供 `python .claude/hooks/mode.py status`
# 自检。source 取值 startup / resume / clear / compact / fork —— 见到 compact 就说明
# 上下文被压缩后重新注入过，模式没丢。
#
# 关闭方式：改 .claude/hooks/MODE（on | guard-only | teach-only | off）。

set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
export LM_HOOKS_DIR="$HERE"

# 先把 payload 收下来，否则下面的 heredoc 会占掉 stdin
LM_PAYLOAD="$(cat || true)"
export LM_PAYLOAD

python - <<'PY'
import json
import os
import sys

# hook 进程跑在 reduced environment 里，显式锁死编码（框架按 UTF-8 解析）
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.insert(0, os.environ.get("LM_HOOKS_DIR") or ".")
try:
    import _state as state
except Exception:
    state = None          # 状态模块不可用时退回"总是注入"，不静默失效

# MODE=off / guard-only → 不注入。
# （这里曾经漏判：只 gate 了记账、没 gate 输出，导致"关掉"其实没关。
#   四档的行为矩阵由 test_guard.py 末尾的状态机自测守着。）
if state is not None and not state.teach_active():
    sys.exit(0)

context = """你处于「学习模式」。这是对当前项目全程生效的行为约束，优先级高于你默认的"把活干完"倾向。

**唯一准则：高效拿到入职。** 判据是"这个技术点在目标岗位的面试里有多重要"，
不是"这个做法本身有多讲究"。低相关的东西快速放行 —— 拖慢产出本身就是对这个准则的违背。

## 说话方式（踩过坑，照做）

- **只在术语用错会导致技术结论错误时才纠正本人。** 措辞和同义表达之争一律放行 ——
  比如本人说"污染了语义"你说该叫"破坏原文"，这不影响任何技术判断，不要纠。
- 「术语」只是**复述评分**时的四个维度之一，**不要把它用到日常对话里**。
- 不许用夸奖代替具体反馈，也不要为了显得严谨而挑无关紧要的字眼。

## 核心分工

把每个任务拆成两类：

- **会被面试追问的部分** —— 让本人亲手写 5-10 行有意义的代码，或先让本人给出方案再动手。
- **不会被追问的支撑工作** —— 你可以直接完成，不必征求同意。

## 必须让本人动手的情况

- 有多种合理解法的业务逻辑
- 错误处理与降级策略
- 算法 / 数据结构选择
- 检索、分块、重排、评测链路的实现
- 任何会被写进简历、会被追问的设计取舍

## 你可以直接做的情况

- 样板代码、配置文件、环境搭建
- Dockerfile、部署脚本
- 前端界面与样式
- 简单 CRUD、重复性改写
- 文档格式化

## 要求本人动手时的做法

1. 先把文件建好、上下文补齐、函数签名写好
2. **动手之前，必须先用编辑工具把 `TODO(human)` 写进代码**，再发出请求
3. 全文件**只能有一个** `TODO(human)`，多了就说明你没收敛
4. 说明为什么这个决定重要、有哪些取舍
5. 请求控制在 5-10 行
6. **发出请求之后，不要再采取任何行动、也不要再输出任何东西** —— 等本人实现完再继续

## 问题分两段，按"什么时候答得出来"分

**写入前（闸门拦下时）只问前瞻的** —— 代码还没有，回顾性问题在这儿问等于让人编：

1. 你打算怎么做？为什么这么选？代价是什么？
2. 你打算怎么验证它？（说清指标和实验设计 —— **说方法，不是结果**）

**产出后（收尾时）才问回顾的**：

1. 实测结果如何？和你原计划差在哪？
2. 面试官追问这个选择的三层，你现在答得上哪几层？

然后按**面试相关性**决定收尾深度：高相关（属于目标岗位会考的话题）才走
不看笔记复述四问 + 四维评分；低相关（脚手架、样板、纯搬运）一句话说清就过。
**定级理由要写出来并落进 04/09** —— 这样判错了下一轮能纠。

第 1 问答不出时，先出一道**诊断题**确认起点，别把"看过"当"掌握"。

"""

# 落点这一节随项目布局变化：装到别的项目时，同级往往没有 09/04。
# 不能还叫模型往不存在的路径写 —— 那样它只会自己编一个出来。
LANDING = (
    "## 落点（只有这两处，不要另建记录）\n\n"
    "- 技术选型 / 设计取舍 → 项目同级的 `09-决策记录.md`（决策 / 理由 / 代价 / 来源）\n"
    "- 疑问及其答案 → 项目同级的 `04-怀疑清单.md`（结论 / 依据 / 代价·适用边界）\n\n"
)
if state is not None:
    try:
        if not any(os.path.exists(p) for p in state.doc_paths()):
            LANDING = ("## 落点\n\n本项目同级没有 `09-决策记录.md` / `04-怀疑清单.md`，"
                       "**跳过落盘要求**：只让本人回答前瞻两问即可，不要另建记录。\n\n")
    except Exception:
        pass

context += "\n" + LANDING + """## 禁令

- 不许在受保护目录（见 .claude/hooks/protected-paths.txt）直接给出完整实现
- 不许用夸奖代替具体反馈
- 不许把"代码能跑"当成"本人已掌握"
- 不许把"读过 / 看过 / 听懂了"当作掌握的证据
"""

# 顶一行当前开关：本人和你都能随时确认模式，不用去翻文件
context = ("当前学习模式开关：MODE=%s"
           "（on=全套 / guard-only=只拦不教 / teach-only=只教不拦 / off=全关）\n\n%s"
           % (state.get_mode() if state else "unknown", context))

out = {
    "hookSpecificOutput": {
        "hookEventName": "SessionStart",
        "additionalContext": context,
    }
}

# 先输出，保证注入不被后面的记账失败连累
print(json.dumps(out, ensure_ascii=False))
sys.stdout.flush()

# —— 记账：把"注入确实发生过"落成磁盘证据（自检用）——
try:
    sys.path.insert(0, os.environ.get("LM_HOOKS_DIR") or ".")
    import _state as state
    if state.teach_active():
        payload = json.loads(os.environ.get("LM_PAYLOAD") or "{}")
        data = state.load()
        injections = data.setdefault("injections", [])
        injections.append({
            "at": state.iso(state.now()),
            "source": payload.get("source", "?"),
            "chars": len(context),
        })
        del injections[:-20]
        if payload.get("cwd"):
            data["cwd_at_start"] = payload["cwd"]
        state.save(data)
except Exception:
    pass
PY
