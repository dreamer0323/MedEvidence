#!/usr/bin/env bash
# Stop hook 的哨兵。
#
# 为什么要有这一层：Stop 每回合都触发一次，而绝大多数回合根本没有教学回合。
# 用纯 bash 判断"有没有开放回合"，没有就直接 exit 0 —— 常见路径完全不启动 Python。
#
# 只有存在 state/open-* 哨兵（说明某个会话正开着回合）时，才把 stdin 上的
# Stop payload 交给 stop.py 处理。
DIR="$(cd "$(dirname "$0")" && pwd)"

# 开关为 off / guard-only 时，收尾强制整条关掉。
# 注意用 `cat ... 2>/dev/null |` 而不是 `< "$DIR/MODE" 2>/dev/null"`：
# bash 从左到右设重定向，`< MODE` 在 MODE 不存在时先报错，那一刻 2>/dev/null
# 还没生效，错误会漏到 stderr（本文件踩过这个坑）。
MODE="$(cat "$DIR/MODE" 2>/dev/null | tr -d '[:space:]')"
case "$MODE" in
  off|guard-only) exit 0 ;;
esac

for f in "$DIR"/state/open-*; do
  if [ -e "$f" ]; then
    exec python "$DIR/stop.py"
  fi
done
exit 0
