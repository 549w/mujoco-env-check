"""平台探测的纯函数与平台知识表。

设计原则：所有判定逻辑都是"接受事实输入"的纯函数（例如 detect_wsl 接受 /proc/version 的文本），
这样可以在任意平台上用合成输入验证判定分支（如在 macOS 上验证 WSL / Blackwell 分支）。
真正读取系统的动作集中在 gather_* 函数里，保持薄封装。
"""

from __future__ import annotations

import os
import platform
import re
import subprocess
import sys
from pathlib import Path

# 当前 mujoco（3.14.x）要求 Python >= 3.10（检测工具自身 3.8 即可运行）
MUJOCO_MIN_PYTHON = (3, 10)

# 各平台合法的 MUJOCO_GL 后端与默认后端
VALID_MUJOCO_GL = {
    "macos": ["cgl", "glfw"],
    "linux": ["egl", "glfw", "glx", "osmesa"],
    "windows": ["glfw", "wgl"],
}
DEFAULT_MUJOCO_GL = {"macos": "cgl", "linux": "glfw", "windows": "glfw"}

_GL_OFF_VALUES = {"disable", "disabled", "off", "false", "0"}
_GL_ON_VALUES = {"enable", "enabled", "on", "true", "1", ""}

# GPU 计算能力 -> 所需最低 CUDA 运行时。仅用于提示"驱动过旧"，不做深度分析。
# Blackwell(sm_100/103/120/121) 需要 CUDA 12.8+；Hopper/Ada 11.8+；Ampere 11.0+；Turing 10.0+
_MIN_CUDA_BY_COMPUTE_CAP = [
    (10.0, "12.8"),
    (9.0, "11.8"),
    (8.9, "11.8"),
    (8.0, "11.0"),
    (7.5, "10.0"),
]


# ---------- 纯函数：平台判定 ----------

def os_key(system_name: str) -> str:
    """platform.system() 的取值 -> 归一化平台名（macos/linux/windows/unknown）。"""
    mapping = {"darwin": "macos", "linux": "linux", "windows": "windows"}
    return mapping.get((system_name or "").lower(), "unknown")


def check_python_version(version_tuple) -> tuple:
    """返回 (是否满足 mujoco 要求, 说明)。"""
    parts = tuple(version_tuple)[:2]
    short = ".".join(str(part) for part in tuple(version_tuple)[:3])
    if parts >= MUJOCO_MIN_PYTHON:
        return True, f"Python {short} satisfies mujoco's requirement (>= 3.10)"
    return False, f"Python {short} is older than mujoco's minimum (3.10); recent mujoco wheels will not install"


def wheel_availability(os_name: str, machine: str) -> tuple:
    """返回 (该平台/架构是否有官方 wheel, 说明)。依据 mujoco 3.11+ 的 wheel 覆盖范围。"""
    machine = (machine or "").lower()
    if os_name == "macos":
        if machine == "arm64":
            return True, "official mujoco wheels exist for macOS arm64"
        return False, ("mujoco >= 3.11 no longer ships macOS x86_64 wheels; pip would attempt a source "
                       "build and most likely fail (an arm64 Python is required)")
    if os_name == "linux":
        if machine in ("x86_64", "amd64", "aarch64", "arm64"):
            return True, f"official mujoco wheels exist for Linux {machine}"
        return False, f"no known mujoco wheels for Linux {machine}"
    if os_name == "windows":
        if machine in ("amd64", "x86_64"):
            return True, "official mujoco wheels exist for Windows amd64"
        return False, f"no known mujoco wheels for Windows {machine}; use an x86_64 Python"
    return False, f"wheel availability unknown for platform '{os_name}/{machine}'"


def detect_wsl(proc_version_text: str, wslg_dir_exists: bool = False, wayland_display: str = "") -> dict:
    """依据 /proc/version 等事实判定 WSL 与 WSLg。WSL2 的内核串含 'microsoft-standard'。"""
    text = (proc_version_text or "").lower()
    if "microsoft" not in text and "wsl" not in text:
        return {"is_wsl": False, "version": None, "wslg": False}
    version = "WSL2" if ("wsl2" in text or "microsoft-standard" in text) else "WSL1(?)"
    return {"is_wsl": True, "version": version, "wslg": bool(wslg_dir_exists) or bool(wayland_display)}


def detect_rosetta(proc_translated: str) -> bool:
    """macOS：x86_64 解释器跑在 Apple Silicon 上（sysctl.proc_translated == 1）。"""
    return (proc_translated or "").strip() == "1"


def classify_mujoco_gl(os_name: str, value) -> tuple:
    """返回 (kind, 说明)，kind ∈ unset|off|on|valid|invalid。"""
    if value is None or value == "":
        default = DEFAULT_MUJOCO_GL.get(os_name)
        if default is None:
            return "unset", "MUJOCO_GL not set"
        return "unset", f"MUJOCO_GL not set; the platform default backend is '{default}'"
    raw = value.strip()
    lowered = raw.lower()
    if lowered in _GL_OFF_VALUES:
        return "off", f"MUJOCO_GL={raw}: rendering is explicitly disabled"
    if lowered in _GL_ON_VALUES:
        return "on", f"MUJOCO_GL={raw}: backend auto-selected"
    valid = VALID_MUJOCO_GL.get(os_name, [])
    if lowered in valid:
        return "valid", f"MUJOCO_GL={raw}: valid backend for {os_name}"
    valid_text = ", ".join(valid) if valid else "unknown"
    return "invalid", (f"MUJOCO_GL={raw} is not a valid backend on {os_name}; "
                       f"valid values: {valid_text} (or disable/enable)")


def describe_context(os_name: str, wsl: dict, container: bool, headless_hints) -> str:
    """一行环境上下文；后续渲染/GPU 结论都要靠它来解释。"""
    if wsl.get("is_wsl"):
        parts = [wsl["version"] + (" with WSLg" if wsl["wslg"] else " without WSLg")]
    elif os_name == "macos":
        parts = ["native macOS"]
    elif os_name == "linux":
        parts = ["native Linux"]
    elif os_name == "windows":
        parts = ["native Windows"]
    else:
        parts = ["unknown platform"]
    if container:
        parts.append("container")
    if "ssh" in headless_hints:
        parts.append("SSH session")
    if "no-display" in headless_hints:
        parts.append("no DISPLAY/WAYLAND_DISPLAY")
    return ", ".join(parts)


def parse_nvidia_smi_csv(text: str) -> list:
    """解析 nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader 的输出。"""
    gpus = []
    for line in (text or "").splitlines():
        parts = [item.strip() for item in line.split(",")]
        if len(parts) < 4 or not parts[0]:
            continue
        gpus.append({
            "name": parts[0],
            "driver_version": parts[1],
            "memory": parts[2],
            "compute_cap": parts[3],
            "compute_cap_value": _to_float(parts[3]),
        })
    return gpus


def parse_nvidia_smi_cuda_version(text: str):
    """从 nvidia-smi 完整输出中提取 'CUDA Version: 12.8' -> (12, 8)。"""
    match = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", text or "")
    if not match:
        return None
    return (int(match.group(1)), int(match.group(2)))


def min_cuda_for_compute_cap(cap_value):
    """计算能力对应的最低 CUDA 版本；未知时返回 None。"""
    if cap_value is None:
        return None
    for threshold, version in _MIN_CUDA_BY_COMPUTE_CAP:
        if cap_value >= threshold:
            return version
    return None


def evaluate_gpu_compat(gpus: list, driver_cuda) -> list:
    """判定每块 GPU 的算力与驱动自带 CUDA 版本是否匹配。

    driver_cuda 形如 (12, 8)。每条结果含 status（ok|warning|unknown）与说明文字。
    """
    items = []
    for gpu in gpus:
        required = min_cuda_for_compute_cap(gpu.get("compute_cap_value"))
        driver_text = ".".join(str(part) for part in driver_cuda) if driver_cuda else None
        item = {
            "name": gpu.get("name"),
            "compute_cap": gpu.get("compute_cap"),
            "required_cuda": required,
            "driver_cuda": driver_text,
            "status": "unknown",
        }
        if required is not None and driver_cuda is not None:
            if tuple(driver_cuda) < tuple(int(part) for part in required.split(".")):
                item["status"] = "warning"
                item["message"] = (f"the driver supports CUDA up to {driver_text}, but compute capability "
                                   f"{gpu.get('compute_cap')} requires CUDA >= {required}")
            else:
                item["status"] = "ok"
        items.append(item)
    return items


# ---------- 系统读取（薄封装，便于在其它平台复用纯函数） ----------

def describe_python_runtime() -> str:
    """当前解释器归属：conda / venv / system。"""
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix and Path(conda_prefix) == Path(sys.prefix):
        return "conda"
    if sys.prefix != getattr(sys, "base_prefix", sys.prefix) or os.environ.get("VIRTUAL_ENV"):
        return "venv"
    return "system"


def gather_headless_hints() -> list:
    """收集"没有图形会话"的线索：SSH 会话、Linux 无 DISPLAY。"""
    hints = []
    if os.environ.get("SSH_CONNECTION") or os.environ.get("SSH_TTY"):
        hints.append("ssh")
    if os_key(platform.system()) == "linux" and not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        hints.append("no-display")
    return hints


def gather_platform_facts() -> dict:
    """一次性收集本机平台事实，供各 check 使用。"""
    os_name = os_key(platform.system())
    wsl = detect_wsl(_read_text("/proc/version"), Path("/mnt/wslg").is_dir(),
                     os.environ.get("WAYLAND_DISPLAY", ""))
    container = (Path("/.dockerenv").exists()
                 or Path("/run/.containerenv").exists()
                 or "docker" in _read_text("/proc/1/cgroup").lower())
    headless_hints = gather_headless_hints()
    rosetta = os_name == "macos" and detect_rosetta(_sysctl("sysctl.proc_translated"))

    return {
        "os": os_name,
        "os_full": f"{platform.system()} {platform.release()}",
        "arch": platform.machine(),
        "python_version": platform.python_version(),
        "python_tuple": tuple(sys.version_info[:3]),
        "python_executable": sys.executable,
        "runtime": describe_python_runtime(),
        "wsl": wsl,
        "container": container,
        "rosetta": rosetta,
        "headless_hints": headless_hints,
        "mujoco_gl": os.environ.get("MUJOCO_GL"),
        "context": describe_context(os_name, wsl, container, headless_hints),
    }


def _read_text(path: str) -> str:
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _sysctl(name: str) -> str:
    try:
        proc = subprocess.run(["sysctl", "-n", name], capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return ""
    return proc.stdout.strip() if proc.returncode == 0 else ""


def _to_float(text):
    try:
        return float(text)
    except (TypeError, ValueError):
        return None
