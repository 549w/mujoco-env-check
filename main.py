#!/usr/bin/env python3
"""MuJoCo 环境检测入口。

职责：按注册表把每个 check 作为独立子进程运行（某项崩溃或挂起不影响整体报告），
聚合结果后打印终端报告，并写入 reports/latest.json 供后续（Agent）读取。
"""

from __future__ import annotations

import argparse
import os
import platform
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from utils import report

REPO_ROOT = Path(__file__).resolve().parent
JSON_PATH = REPO_ROOT / "reports" / "latest.json"

# 检查注册表：新增一个检查 = 新建 checks/<id>_check.py + 在这里加一行
CHECKS = [
    {"id": "system", "module": "checks.system_check", "title": "System information", "level": 0, "timeout": 10},
    {"id": "mujoco", "module": "checks.mujoco_check", "title": "MuJoCo import", "level": 1, "timeout": 20},
    {"id": "simulation", "module": "checks.simulation_check", "title": "Simulation", "level": 2, "timeout": 60},
    {"id": "render", "module": "checks.render_check", "title": "Rendering", "level": 3, "timeout": 90},
    {"id": "gpu", "module": "checks.gpu_check", "title": "GPU acceleration", "level": 4, "timeout": 15},
]

# 依赖 mujoco 可用的检查：导入失败时直接标记 SKIPPED，避免次级报错掩盖根因
DEPENDENT_ON_MUJOCO = ("simulation", "render")


def _kill(proc: subprocess.Popen) -> None:
    if os.name == "posix":
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            return
        except (ProcessLookupError, PermissionError):
            pass
    proc.kill()


def _tail(text: str, lines: int = 3) -> str:
    parts = [line for line in (text or "").strip().splitlines() if line.strip()]
    return " | ".join(parts[-lines:])


def run_check(spec: dict, timeout: float) -> report.CheckResult:
    """以子进程运行单个检查，返回其结果；崩溃/超时/无输出都会被翻译成可读的 ERROR。"""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(REPO_ROOT), env.get("PYTHONPATH", "")]))
    env["PYTHONIOENCODING"] = "utf-8"
    cmd = [sys.executable, "-m", spec["module"]]
    popen_kwargs = {"start_new_session": True} if os.name == "posix" else {}

    start = time.monotonic()
    proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT), env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, encoding="utf-8", errors="replace", **popen_kwargs)
    timed_out = False
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        _kill(proc)
        stdout, stderr = proc.communicate()
    elapsed_ms = int((time.monotonic() - start) * 1000)

    if timed_out:
        evidence = [f"stderr tail: {_tail(stderr)}"] if stderr.strip() else []
        return report.result(spec["id"], spec["title"], spec["level"], report.ERROR,
                             f"check timed out after {timeout:g}s and was killed",
                             evidence=evidence, duration_ms=elapsed_ms,
                             suggestions=[f"re-run with a larger --timeout, or run it directly: "
                                          f"python3 -m {spec['module']}"])

    parsed = report.parse_child_stdout(stdout)
    if parsed is None:
        evidence = [f"exit code: {proc.returncode}"]
        for label, text in (("stderr", stderr), ("stdout", stdout)):
            tail = _tail(text)
            if tail:
                evidence.append(f"{label} tail: {tail}")
        return report.result(spec["id"], spec["title"], spec["level"], report.ERROR,
                             "check crashed before producing a result",
                             evidence=evidence, duration_ms=elapsed_ms,
                             suggestions=[f"run it directly to see the full output: python3 -m {spec['module']}"])

    try:
        return report.CheckResult(**parsed)
    except TypeError:
        return report.result(spec["id"], spec["title"], spec["level"], report.ERROR,
                             "check produced an unreadable result",
                             evidence=[str(parsed)[:200]], duration_ms=elapsed_ms)


def system_view(system_result) -> dict:
    """报告头部的 System 区块：优先用 system check 的结果，缺失时退回 platform 库。"""
    if system_result is not None:
        details = system_result.details
        return {"os": details.get("os"), "arch": details.get("arch"),
                "python": details.get("python"), "context": details.get("context")}
    return {"os": f"{platform.system()} {platform.release()}", "arch": platform.machine(),
            "python": platform.python_version(), "context": None}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify that the MuJoCo environment on this machine is usable.")
    parser.add_argument("--only", action="append", choices=[c["id"] for c in CHECKS],
                        metavar="CHECK", help="run only the given check (repeatable)")
    parser.add_argument("--timeout", type=float, default=None,
                        help="override the per-check timeout in seconds (debugging)")
    args = parser.parse_args()

    selected = [spec for spec in CHECKS if not args.only or spec["id"] in args.only]
    results = {}
    for spec in selected:
        if spec["id"] in DEPENDENT_ON_MUJOCO:
            mujoco_result = results.get("mujoco")
            if mujoco_result is not None and mujoco_result.status in (report.FAIL, report.ERROR):
                results[spec["id"]] = report.result(
                    spec["id"], spec["title"], spec["level"], report.SKIPPED,
                    "mujoco is not importable (see the 'MuJoCo import' check)",
                    suggestions=["fix the import failure first"])
                continue
        timeout = args.timeout if args.timeout is not None else spec["timeout"]
        print(f"running: {spec['id']} ...", file=sys.stderr, flush=True)
        results[spec["id"]] = run_check(spec, timeout)

    ordered = [results[spec["id"]] for spec in CHECKS if spec["id"] in results]
    summary = report.compute_summary(results)
    system_result = results.get("system")
    view = system_view(system_result)

    payload = {
        "schema_version": 1,
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "system": system_result.details if system_result is not None else view,
        "checks": {res.check_id: res.to_dict() for res in ordered},
        "summary": summary,
    }

    print(report.render_text(view, ordered, summary, str(JSON_PATH)))
    try:
        report.write_json(JSON_PATH, payload)
    except OSError as exc:
        print(f"WARNING: could not write {JSON_PATH}: {exc}", file=sys.stderr)

    return summary["exit_code"]


if __name__ == "__main__":
    sys.exit(main())
