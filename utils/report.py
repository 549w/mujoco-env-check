"""结果模型与报告渲染。

check 子进程与 main.py 共用这个模块：
- 子进程侧：result() 构造结果，run_check_cli() 作为统一入口并输出哨兵 JSON
- 父进程侧：parse_child_stdout() 解析子进程输出，compute_summary()/render_text()/write_json() 聚合输出
"""

from __future__ import annotations

import json
import time
import traceback
from dataclasses import asdict, dataclass, field

PASS = "PASS"
WARNING = "WARNING"
FAIL = "FAIL"
SKIPPED = "SKIPPED"
ERROR = "ERROR"

# 子进程 -> 父进程协议：stdout 中的哨兵行（容忍 GL 等库往 stdout 打日志）
SENTINEL = "MUJOCO_ENV_CHECK_RESULT:"

# 决定 "Core MuJoCo environment" 汇总结论的检查项
CORE_CHECK_IDS = ("system", "mujoco", "simulation")

_SEVERITY = {PASS: 0, SKIPPED: 1, WARNING: 2, ERROR: 3, FAIL: 4}


@dataclass
class CheckResult:
    """单个检查的完整结果，字段与 reports/latest.json 中的结构一一对应。"""

    check_id: str
    title: str
    level: int
    status: str
    message: str = ""
    details: dict = field(default_factory=dict)
    evidence: list = field(default_factory=list)
    suggestions: list = field(default_factory=list)
    duration_ms: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def result(check_id, title, level, status, message="", details=None, evidence=None,
           suggestions=None, duration_ms=0) -> CheckResult:
    return CheckResult(
        check_id=check_id,
        title=title,
        level=level,
        status=status,
        message=message,
        details=details or {},
        evidence=list(evidence or []),
        suggestions=list(suggestions or []),
        duration_ms=duration_ms,
    )


def worst_status(*statuses: str) -> str:
    """取最严重的状态（FAIL > ERROR > WARNING > SKIPPED > PASS）。"""
    known = [status for status in statuses if status in _SEVERITY]
    if not known:
        return PASS
    return max(known, key=lambda status: _SEVERITY[status])


def run_check_cli(check_id: str, title: str, level: int, run_func) -> None:
    """check 脚本的统一入口：执行检测、兜底捕获崩溃、输出哨兵 JSON。"""
    start = time.monotonic()
    try:
        res = run_func()
    except Exception as exc:  # check 自身没接住的异常 = 该检查崩溃
        tb_lines = traceback.format_exc().strip().splitlines()
        res = result(check_id, title, level, ERROR,
                     f"unhandled {type(exc).__name__}: {exc}",
                     evidence=tb_lines[-6:])
    if not res.duration_ms:
        res.duration_ms = int((time.monotonic() - start) * 1000)
    print(SENTINEL + json.dumps(res.to_dict(), ensure_ascii=True))


def parse_child_stdout(stdout_text: str):
    """从子进程 stdout 中解析最后一条哨兵行，返回 dict 或 None。"""
    for line in reversed((stdout_text or "").splitlines()):
        line = line.strip()
        if line.startswith(SENTINEL):
            try:
                return json.loads(line[len(SENTINEL):])
            except json.JSONDecodeError:
                return None
    return None


def compute_summary(results: dict) -> dict:
    """results: check_id -> CheckResult（未执行的检查不在字典中）。"""
    executed_core = [results[cid] for cid in CORE_CHECK_IDS if cid in results]
    if not executed_core:
        core = "n/a"
    elif any(r.status in (FAIL, ERROR) for r in executed_core):
        core = FAIL
    else:
        core = PASS

    render = results.get("render")
    gpu = results.get("gpu")
    return {
        "core": core,
        "rendering": render.status if render else "n/a",
        "gpu": gpu.status if gpu else "n/a",
        "exit_code": 1 if core == FAIL else 0,
    }


def render_text(system_view: dict, results: list, summary: dict, json_path: str) -> str:
    """渲染终端报告。输出保持纯 ASCII，避免 Windows 控制台编码问题。"""
    bar = "=" * 48
    lines = [bar, "MuJoCo Environment Report", bar, "", "System", "------"]
    lines.append(f"OS: {system_view.get('os') or 'unknown'}")
    lines.append(f"Arch: {system_view.get('arch') or 'unknown'}")
    lines.append(f"Python: {system_view.get('python') or 'unknown'}")
    if system_view.get("context"):
        lines.append(f"Context: {system_view['context']}")

    lines += ["", "Checks", "------"]
    for res in results:
        lines.append("")
        lines.append(f"[{res.status}] {res.title} (Level {res.level})")
        if res.message:
            lines.append(f"      {res.message}")
        for item in res.evidence:
            lines.append(f"      - {item}")
        for tip in res.suggestions:
            lines.append(f"      Suggest: {tip}")

    lines += ["", "Summary", "-------", ""]
    lines.append(f"Core MuJoCo environment: {summary['core']}")
    lines.append(f"Rendering: {summary['rendering']}")
    lines.append(f"GPU: {summary['gpu']}")
    lines += ["", f"Full report: {json_path}", bar]
    return "\n".join(lines)


def write_json(path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=True)
        fh.write("\n")
