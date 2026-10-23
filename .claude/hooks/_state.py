#!/usr/bin/env python
"""会话状态：教学回合的存储与判定。

被 guard.py / prompt.py / stop.py / mode.py / learning-mode.sh 共用。

设计要点（对着 gateguard 的 #2049「永久 deny」事故逐条防）：
  · 状态按 CLAUDE_CODE_SESSION_ID 分文件 —— 会话键由框架提供，天然收敛
  · 原子写（临时文件 + os.replace）—— 防并发写坏
  · TTL：超时的回合视为陈旧并清理 —— 防状态卡死导致永久阻断
  · 项目里找不到 09/04 时，不把"记录落盘"当作放行条件 —— 防止装到没有
    这两个文件的项目上把闸门锁死（对齐 guard.py 的 fail-open 哲学）
  · 所有拦截 / 放行 / 放弃都记账 —— 让绕过可审计，不静默
"""
import hashlib
import json
import os
import re
import tempfile
from datetime import datetime, timedelta

HERE = os.path.dirname(os.path.abspath(__file__))
# 状态目录可用环境变量覆盖 —— 自测靠它隔离，绝不能写进生产状态。
# （踩过的坑：test_guard.py 直接跑真 guard.py，而 CLAUDE_CODE_SESSION_ID 会从
#  父 shell 继承，于是自测把一条假回合写进了真实会话的状态里。）
STATE_DIR = os.environ.get("HOOKS_STATE_DIR") or os.path.join(HERE, "state")
MODE_FILE = os.path.join(HERE, "MODE")
GUARD_OFF = os.path.join(HERE, "GUARD-OFF")   # 弃用别名，保留兼容

TTL_HOURS = 12             # 回合超过这么久视为陈旧
MAX_STOP_BLOCKS = 2        # 每回合最多阻断收尾几次

MODE_VALUES = ("on", "guard-only", "teach-only", "off")

DOC_09 = "09-决策记录.md"
DOC_04 = "04-怀疑清单.md"


# ---------------------------------------------------------------- 时间

def now() -> datetime:
    return datetime.now()


def iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%S")


def parse(s):
    try:
        return datetime.strptime(s, "%Y-%m-%dT%H:%M:%S")
    except Exception:
        return None


# ---------------------------------------------------------------- 开关

def get_mode() -> str:
    """读单一开关。文件缺失或值非法都按 on。"""
    try:
        with open(MODE_FILE, encoding="utf-8") as fh:
            v = fh.read().strip().lower()
    except OSError:
        return "on"
    return v if v in MODE_VALUES else "on"


def set_mode(value: str) -> None:
    v = (value or "").strip().lower()
    if v not in MODE_VALUES:
        raise ValueError("非法值 %r，只接受 %s" % (value, " | ".join(MODE_VALUES)))
    os.makedirs(os.path.dirname(MODE_FILE), exist_ok=True)
    with open(MODE_FILE, "w", encoding="utf-8") as fh:
        fh.write(v + "\n")


def guard_off_present() -> bool:
    return os.path.exists(GUARD_OFF)


def guard_active() -> bool:
    """闸门是否生效。GUARD-OFF 是弃用的粗粒度别名，仍尊重。"""
    if get_mode() in ("off", "teach-only"):
        return False
    return not guard_off_present()


def teach_active() -> bool:
    """是否注入教学文本 + 每轮提醒 + 收尾强制。"""
    return get_mode() in ("on", "teach-only")


def require_user_turn() -> bool:
    """放行是否要求"回合开启后本人说过话"。

    guard-only 下每轮提醒与收尾强制都关了，拿不到用户输入时间戳，
    因此降级为只看"记录是否落盘"——避免把闸门永久锁死。
    """
    return get_mode() == "on"


# ---------------------------------------------------------------- 路径

def session_id() -> str:
    return (os.environ.get("CLAUDE_CODE_SESSION_ID")
            or os.environ.get("CLAUDE_SESSION_ID")
            or "unknown")


def _safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", s or "unknown")


_MSYS_RE = re.compile(r"^/([A-Za-z])/(.*)$")


def norm_path(p: str) -> str:
    """把 Git Bash 的 /d/dev/x 归一成 D:/dev/x。

    不归一化会**静默全放行**：Windows 的 Python 把 /d/dev/x 解析成
    <当前盘>:\\d\\dev\\x，于是读不到 protected-paths.txt，守卫直接放行。
    从 Git Bash 里拿 `$PWD` 传给 apply.sh 或直接调用时就会踩到。
    """
    s = (p or "").strip().strip('"').strip("'").replace("\\", "/")
    m = _MSYS_RE.match(s)
    return (m.group(1).upper() + ":/" + m.group(2)) if m else s


def project_root() -> str:
    r = os.environ.get("CLAUDE_PROJECT_DIR")
    if r and r.strip():
        return norm_path(r)
    # hooks 目录在 <root>/.claude/hooks，上两级即 root
    return os.path.dirname(os.path.dirname(HERE))


def doc_paths(root: str = None):
    """09 / 04 两个落点文件的绝对路径（在项目根的上一级，即 plan/ 下）。"""
    base = os.path.dirname(os.path.abspath((root or project_root()).rstrip("/\\")))
    return [os.path.join(base, DOC_09), os.path.join(base, DOC_04)]


# ---------------------------------------------------------------- 读写

def state_path(sid: str = None) -> str:
    return os.path.join(STATE_DIR, "session-%s.json" % _safe(sid or session_id()))


def _blank(sid: str = None) -> dict:
    return {
        "session_id": sid or session_id(),
        "started_at": iso(now()),
        "cwd_at_start": "",
        "injections": [],
        # 用户输入用单调计数器记录，而不是靠时间戳比大小。
        # 原因：判断"本人开口是否发生在回合开启之后"本质是事件排序问题，
        # 而秒级时间戳分不开同一秒内的两件事（自测第 4 步就是这么挂的）。
        "user_turn_count": 0,
        "last_user_turn": "",
        "round": None,
        "counters": {"blocks": 0, "grants": 0, "stop_blocks": 0, "stop_block_limits": 0},
    }


def load(sid: str = None) -> dict:
    try:
        with open(state_path(sid), encoding="utf-8") as fh:
            d = json.load(fh)
        if isinstance(d, dict):
            base = _blank(sid)
            base.update(d)
            base.setdefault("counters", {})
            return base
    except Exception:
        pass
    return _blank(sid)


def save(data: dict, sid: str = None) -> None:
    """原子写：临时文件 + os.replace，避免并发/中断写坏状态。"""
    os.makedirs(STATE_DIR, exist_ok=True)
    p = state_path(sid or data.get("session_id"))
    fd, tmp = tempfile.mkstemp(dir=STATE_DIR, prefix=".tmp-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
        os.replace(tmp, p)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def all_sessions():
    """列出所有会话状态文件路径，按修改时间新→旧。"""
    try:
        names = [n for n in os.listdir(STATE_DIR)
                 if n.startswith("session-") and n.endswith(".json")]
    except OSError:
        return []
    paths = [os.path.join(STATE_DIR, n) for n in names]
    return sorted(paths, key=lambda p: os.path.getmtime(p), reverse=True)


def file_hash(path: str) -> str:
    try:
        with open(path, "rb") as fh:
            return hashlib.sha256(fh.read()).hexdigest()[:16]
    except OSError:
        return ""


# ---------------------------------------------------------------- 哨兵

def sentinel_path(sid: str = None) -> str:
    return os.path.join(STATE_DIR, "open-%s" % _safe(sid or session_id()))


def mark_open(sid: str = None) -> None:
    try:
        os.makedirs(STATE_DIR, exist_ok=True)
        open(sentinel_path(sid), "w").close()
    except OSError:
        pass


def clear_open(sid: str = None) -> None:
    try:
        os.unlink(sentinel_path(sid))
    except OSError:
        pass


# ---------------------------------------------------------------- 回合

def round_is_stale(rnd) -> bool:
    if not rnd:
        return True
    t = parse(rnd.get("opened_at", ""))
    return t is None or (now() - t) > timedelta(hours=TTL_HOURS)


def open_round(target: str, root: str = None) -> dict:
    """开一个教学回合。返回回合字典。"""
    root = root or project_root()
    docs = doc_paths(root)
    data = load()
    rnd = {
        "state": "blocking",
        "opened_at": iso(now()),
        "target": target,
        "docs_available": any(os.path.exists(p) for p in docs),
        "baseline": {p: file_hash(p) for p in docs},
        "released_at": None,
        "release_baseline": None,
        "stop_blocks": 0,
        # 事件排序用计数，不用时钟（见 _blank 里的说明）
        "turns_at_open": data.get("user_turn_count", 0),
        "turns_at_release": None,
    }
    data["round"] = rnd
    data["counters"]["blocks"] = data["counters"].get("blocks", 0) + 1
    save(data)
    mark_open()
    return rnd


def close_round(data: dict, reason: str) -> None:
    rnd = data.get("round") or {}
    if rnd:
        rnd["closed_reason"] = reason
        rnd["closed_at"] = iso(now())
        data.setdefault("history", []).append(rnd)
        data["history"] = data["history"][-20:]
    data["round"] = None
    save(data)
    clear_open()


def release_ready(rnd: dict, data: dict, root: str = None):
    """判断 guard 是否该放行。返回 (ok, missing 列表)。

    两条同时成立：
      1. 09 或 04 出现本回合的新增记录（项目里根本没有这两个文件时跳过此条）
      2. 回合开启后至少有一条真实用户输入 —— 模型无法伪造，这是防自放行的关键
    """
    root = root or project_root()
    missing = []

    if rnd.get("docs_available"):
        base = rnd.get("baseline") or {}
        changed = [p for p in doc_paths(root) if file_hash(p) != base.get(p, "")]
        if not changed:
            missing.append("09-决策记录.md / 04-怀疑清单.md 里还没有本回合的新增记录")

    if require_user_turn():
        if (data.get("user_turn_count") or 0) <= (rnd.get("turns_at_open") or 0):
            missing.append("本人还没开口（回合开始后必须有一条真实用户输入，不能由你代答）")

    return (not missing), missing


def release_round(rnd: dict, data: dict, root: str = None) -> None:
    """放行：回合转入 pending_review，并记下这一时刻的落点哈希。"""
    root = root or project_root()
    rnd["state"] = "pending_review"
    rnd["released_at"] = iso(now())
    rnd["release_baseline"] = {p: file_hash(p) for p in doc_paths(root)}
    rnd["turns_at_release"] = data.get("user_turn_count", 0)
    data["counters"]["grants"] = data["counters"].get("grants", 0) + 1
    save(data)


def review_done(rnd: dict, root: str = None) -> bool:
    """放行之后落点又变了 —— 视为复述/评分的结论已落盘。"""
    root = root or project_root()
    rb = rnd.get("release_baseline") or {}
    return any(file_hash(p) != rb.get(p, "") for p in doc_paths(root))


def spoke_after_release(data: dict, rnd: dict) -> bool:
    """放行之后本人是否开过口（= 复述发生了）。同样用计数而非时钟。"""
    at = rnd.get("turns_at_release")
    if at is None:
        return False
    return (data.get("user_turn_count") or 0) > at


# ---------------------------------------------------------------- 文案

# 问题按"什么时候答得出来"分两段。
#
# 这是一次真实的教训：原来前段（写入前的闸门）就问"我怎么验证它是对的？"和
# "追问三层我答得上哪层"—— 那两问是**回顾性**的，代码还没写，只能编。
# 前段只能问前瞻的（打算怎么做、打算怎么验证）；回顾性的归后段（Stop）。
FRONT_QUESTIONS = (
    "动手之前先说清楚（本人说，不要替他答）：\n"
    "  1. 你打算怎么做？为什么这么选？代价是什么？\n"
    "  2. 你打算怎么验证它？（说清指标和实验设计 —— 说方法，不是结果）\n"
)

BACK_QUESTIONS = (
    "这一轮已经产出了代码，现在收尾（本人答，不要替他答）：\n"
    "  1. 实测结果如何？和你原计划差在哪？\n"
    "  2. 面试官追问这个选择的三层，你现在答得上哪几层、答不上哪一层？\n"
)

# 收尾深度按"这个技术点在目标岗位面试里有多重要"分档。
BACK_DEPTH = (
    "\n再按面试相关性决定深度（唯一准则是高效拿到入职，不是把每处都做成教案）：\n"
    "  · 高相关 —— 属于目标岗位面试会考的话题 → 再让本人不看笔记复述四问\n"
    "    （它解决什么 / 怎么工作 / 什么时候失败 / 怎么迁移），\n"
    "    按准确性 / 结构 / 术语 / 迁移各 0-4 打分，结论落进 04 或 09。\n"
    "  · 低相关 —— 脚手架、样板、纯搬运 → 一句话说清就过，别拖。\n"
    "  定级理由要写出来并落进 04/09，这样判错了下一轮能纠。\n"
)


def routing(root: str = None) -> str:
    """阻断文案里的"落点"段。

    随项目实际布局变化：装到别的项目时，同级往往没有 09/04，
    这时不能还叫模型往不存在的路径写 —— 那样它只会编一个出来。
    """
    have = [os.path.basename(p) for p in doc_paths(root) if os.path.exists(p)]
    if not have:
        return ("落点：本项目同级没有配置 09/04 落点文件 —— **跳过落盘要求**，"
                "只让本人回答前瞻两问即可。\n\n")
    return (
        "落点：\n"
        "  · 第 1 问 → 项目同级的 09-决策记录.md（决策 / 理由 / 代价 / 来源）\n"
        "  · 第 2 问（打算怎么验证）也记进 09 —— 收尾时要拿它对照实测结果\n"
        "  · 若本人答不出第 1 问，先出一道诊断题确认起点，别把「看过」当「掌握」\n\n"
    )
