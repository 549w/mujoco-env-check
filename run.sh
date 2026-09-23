#!/bin/sh
# MuJoCo 环境检测入口（macOS / Linux / WSL / Git-Bash）。
# 只负责三件事：找到可用的 Python、做最基本的版本预检、把参数透传给 main.py 并保留退出码。
set -u

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

PY=""
for candidate in python3 python; do
    if command -v "$candidate" >/dev/null 2>&1; then
        # 检测工具自身需要 Python >= 3.8（被检测的 mujoco 环境要求更高，由检查报告给出）
        if "$candidate" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 8) else 1)' >/dev/null 2>&1; then
            PY="$candidate"
            break
        fi
    fi
done

if [ -z "$PY" ]; then
    echo "ERROR: no usable Python found (the checker itself needs Python >= 3.8)." >&2
    echo "       Install Python 3.10+ (required by mujoco), then re-run ./run.sh" >&2
    exit 2
fi

exec "$PY" "$SCRIPT_DIR/main.py" "$@"
