"""Level 1：mujoco Python 绑定。

只验证"包能加载、版本能取到"。注意：mujoco 在 import 阶段就会读取 MUJOCO_GL 并初始化
GL 后端上下文，因此非法后端会在 import 时抛 RuntimeError —— 这里要把这类错误翻译成人话。
"""

from __future__ import annotations

from utils import platform_info, report

CHECK_ID = "mujoco"
TITLE = "MuJoCo import"
LEVEL = 1


def _gl_suggestions(os_name: str) -> list:
    valid = ", ".join(platform_info.VALID_MUJOCO_GL.get(os_name, []))
    return [
        f"the MUJOCO_GL variable selects the GL backend and is validated at import time; valid on {os_name}: {valid}",
        "unset MUJOCO_GL to fall back to the platform default backend",
    ]


def _numpy_version() -> str:
    try:
        import numpy
    except Exception:
        return "unavailable"
    return getattr(numpy, "__version__", "unknown")


def _safe_call(func):
    try:
        return func()
    except Exception:
        return None


def run() -> report.CheckResult:
    facts = platform_info.gather_platform_facts()

    try:
        import mujoco
    except ImportError as exc:
        suggestions = ["pip install mujoco"]
        if not platform_info.check_python_version(facts["python_tuple"])[0]:
            suggestions.append("upgrade Python first: mujoco requires Python >= 3.10")
        wheel_ok, wheel_msg = platform_info.wheel_availability(facts["os"], facts["arch"])
        if not wheel_ok:
            suggestions.append(wheel_msg)
        return report.result(CHECK_ID, TITLE, LEVEL, report.FAIL,
                             f"mujoco package not found: {exc}",
                             details={"os": facts["os"], "arch": facts["arch"]},
                             evidence=[f"ImportError: {exc}"],
                             suggestions=suggestions)
    except RuntimeError as exc:
        # 典型来源：MUJOCO_GL 非法 / GL 后端上下文初始化失败
        return report.result(CHECK_ID, TITLE, LEVEL, report.FAIL,
                             f"mujoco import failed: {exc}",
                             details={"os": facts["os"], "mujoco_gl": facts["mujoco_gl"]},
                             evidence=[f"RuntimeError: {exc}"],
                             suggestions=_gl_suggestions(facts["os"]))
    except Exception as exc:
        return report.result(CHECK_ID, TITLE, LEVEL, report.FAIL,
                             f"mujoco import raised {type(exc).__name__}: {exc}",
                             evidence=[f"{type(exc).__name__}: {exc}"],
                             suggestions=["reinstall the package cleanly: pip install --force-reinstall mujoco"])

    package_version = getattr(mujoco, "__version__", "unknown")
    runtime_version = _safe_call(mujoco.mj_versionString)
    header_version = getattr(mujoco, "mjVERSION_HEADER", None)
    numpy_version = _numpy_version()

    evidence = [
        f"package version: {package_version}",
        f"runtime version (mj_versionString): {runtime_version or 'unavailable'}",
        f"header version (mjVERSION_HEADER): {header_version if header_version is not None else 'unavailable'}",
        f"numpy version: {numpy_version}",
    ]
    details = {
        "version": package_version,
        "runtime_version": runtime_version,
        "header_version": header_version,
        "numpy_version": numpy_version,
    }

    mismatched = (runtime_version and package_version != "unknown"
                  and str(runtime_version) != str(package_version))
    if mismatched:
        return report.result(CHECK_ID, TITLE, LEVEL, report.WARNING,
                             f"mujoco imported, but package ({package_version}) and runtime library "
                             f"({runtime_version}) versions differ",
                             details=details, evidence=evidence,
                             suggestions=["this usually means a mixed or source install; "
                                          "reinstall cleanly: pip install --force-reinstall mujoco"])

    return report.result(CHECK_ID, TITLE, LEVEL, report.PASS,
                         f"mujoco {package_version} imported successfully",
                         details=details, evidence=evidence)


if __name__ == "__main__":
    report.run_check_cli(CHECK_ID, TITLE, LEVEL, run)
