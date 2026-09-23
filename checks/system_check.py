"""Level 0：系统信息与 Python 运行环境。

这里做"平台正确性"的第一道把关：Python 版本门禁、wheel 覆盖范围、macOS Rosetta、
MUJOCO_GL 合法性、以及 WSL / SSH / 无显示等环境上下文（供后续检查解释结论用）。
"""

from __future__ import annotations

from utils import platform_info, report

CHECK_ID = "system"
TITLE = "System information"
LEVEL = 0


def run() -> report.CheckResult:
    facts = platform_info.gather_platform_facts()
    status = report.PASS
    suggestions = []
    evidence = [
        f"{facts['os_full']} / {facts['arch']}",
        f"Python {facts['python_version']} ({facts['runtime']})",
        f"interpreter: {facts['python_executable']}",
        f"environment context: {facts['context']}",
    ]

    python_ok, python_msg = platform_info.check_python_version(facts["python_tuple"])
    evidence.append(python_msg)
    if not python_ok:
        status = report.worst_status(status, report.FAIL)
        suggestions.append("install Python 3.10+ (e.g. create a venv with python3.12) and run the checker with it")

    wheel_ok, wheel_msg = platform_info.wheel_availability(facts["os"], facts["arch"])
    evidence.append(wheel_msg)
    if not wheel_ok:
        status = report.worst_status(status, report.WARNING)
        suggestions.append("do not rely on 'pip install mujoco' on this platform/arch combination (see note above)")

    if facts["rosetta"]:
        status = report.worst_status(status, report.WARNING)
        evidence.append("running under Rosetta 2: the interpreter is x86_64 on Apple Silicon")
        suggestions.append("use a native arm64 Python; mujoco refuses to import under Rosetta 2")

    gl_kind, gl_msg = platform_info.classify_mujoco_gl(facts["os"], facts["mujoco_gl"])
    evidence.append(gl_msg)
    if gl_kind == "invalid":
        status = report.worst_status(status, report.FAIL)
        suggestions.append("unset MUJOCO_GL (or set a valid backend): an invalid value makes 'import mujoco' fail")

    message = (f"{facts['os_full']}, {facts['arch']}, "
               f"Python {facts['python_version']} ({facts['runtime']})")
    details = {
        "os": facts["os_full"],
        "arch": facts["arch"],
        "python": facts["python_version"],
        "context": facts["context"],
        "runtime": facts["runtime"],
        "python_executable": facts["python_executable"],
        "python_ok": python_ok,
        "wheel_ok": wheel_ok,
        "wheel_note": wheel_msg,
        "rosetta": facts["rosetta"],
        "container": facts["container"],
        "wsl": facts["wsl"],
        "headless_hints": facts["headless_hints"],
        "mujoco_gl": facts["mujoco_gl"],
        "mujoco_gl_kind": gl_kind,
    }
    return report.result(CHECK_ID, TITLE, LEVEL, status, message,
                         details=details, evidence=evidence, suggestions=suggestions)


if __name__ == "__main__":
    report.run_check_cli(CHECK_ID, TITLE, LEVEL, run)
