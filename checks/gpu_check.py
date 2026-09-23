"""Level 4：GPU 信息（可选，永远只是信息项）。

核心 mujoco 包不需要 CUDA；CUDA 只与 MJX（jax / mujoco_mjx）相关。
本项只做三件事：报告 GPU 与驱动、提示明显的驱动/CUDA 不匹配（如 Blackwell + 老驱动）、
在 WSL2 里提示驱动透传的注意事项。任何结论都不影响整体判定。
"""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
from pathlib import Path

from utils import platform_info, report

CHECK_ID = "gpu"
TITLE = "GPU acceleration"
LEVEL = 4

SMI_QUERY = "name,driver_version,memory.total,compute_cap"
SMI_TIMEOUT = 8


def _mjx_packages() -> list:
    """只探测是否存在，不真正 import（避免引入 jax 的初始化开销与副作用）。"""
    found = []
    for name in ("jax", "jaxlib", "mujoco_mjx", "mujoco_warp"):
        try:
            if importlib.util.find_spec(name) is not None:
                found.append(name)
        except Exception:
            continue
    return found


def _run_smi(cmd: list):
    """返回 (stdout 或 None, 错误摘要)。"""
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=SMI_TIMEOUT)
    except (OSError, subprocess.SubprocessError) as exc:
        return None, f"{type(exc).__name__}: {exc}"
    if proc.returncode != 0:
        lines = [line for line in (proc.stderr or proc.stdout or "").strip().splitlines() if line.strip()]
        return None, lines[-1] if lines else f"exit code {proc.returncode}"
    return proc.stdout, ""


def _linux_driver_libs() -> list:
    """标准库路径下的 libcuda（WSL2 内出现它通常意味着误装了 Linux 驱动）。"""
    found = []
    for base in ("/usr/lib/x86_64-linux-gnu", "/usr/lib/aarch64-linux-gnu"):
        found.extend(str(path) for path in Path(base).glob("libcuda.so*"))
    return found


def run() -> report.CheckResult:
    facts = platform_info.gather_platform_facts()
    os_name = facts["os"]
    mjx = _mjx_packages()
    mjx_text = ", ".join(mjx) if mjx else "none"
    base_evidence = [f"MJX-related packages installed: {mjx_text}"]

    if os_name == "macos":
        return report.result(CHECK_ID, TITLE, LEVEL, report.SKIPPED,
                             "Not applicable: macOS has no CUDA path (Apple GPU; graphics is handled by CGL)",
                             details={"os": os_name, "mjx_packages": mjx},
                             evidence=base_evidence,
                             suggestions=["CUDA / MJX acceleration only applies to Linux and WSL2"])
    if os_name not in ("linux", "windows"):
        return report.result(CHECK_ID, TITLE, LEVEL, report.SKIPPED,
                             f"Not applicable on this platform ({os_name})",
                             details={"os": os_name, "mjx_packages": mjx},
                             evidence=base_evidence)

    smi = shutil.which("nvidia-smi")
    if not smi:
        evidence = list(base_evidence)
        if os_name == "linux" and Path("/dev/dri").exists():
            evidence.append("/dev/dri exists: a non-NVIDIA GPU may be present, but CUDA does not apply to it")
        return report.result(CHECK_ID, TITLE, LEVEL, report.SKIPPED,
                             "no NVIDIA tooling found (nvidia-smi is not on PATH)",
                             details={"os": os_name, "mjx_packages": mjx, "is_wsl": facts["wsl"]["is_wsl"]},
                             evidence=evidence)

    stdout, error = _run_smi([smi, f"--query-gpu={SMI_QUERY}", "--format=csv,noheader"])
    if stdout is None:
        return report.result(CHECK_ID, TITLE, LEVEL, report.SKIPPED,
                             "nvidia-smi exists but could not query the GPU; GPU/driver state unknown "
                             "(core mujoco does not need CUDA)",
                             details={"os": os_name, "mjx_packages": mjx},
                             evidence=base_evidence + [f"nvidia-smi: {error}"],
                             suggestions=["if you expect usable CUDA, check the NVIDIA driver installation"])

    gpus = platform_info.parse_nvidia_smi_csv(stdout)
    if not gpus:
        return report.result(CHECK_ID, TITLE, LEVEL, report.SKIPPED,
                             "nvidia-smi returned no GPUs",
                             details={"os": os_name, "mjx_packages": mjx}, evidence=base_evidence)

    full_output, _ = _run_smi([smi])
    driver_cuda = platform_info.parse_nvidia_smi_cuda_version(full_output or "")
    compat = platform_info.evaluate_gpu_compat(gpus, driver_cuda)

    status = report.PASS
    suggestions = []
    evidence = list(base_evidence)
    for gpu in gpus:
        evidence.append(f"GPU: {gpu['name']} | driver {gpu['driver_version']} | "
                        f"{gpu['memory']} | compute capability {gpu['compute_cap']}")
    evidence.append(f"driver supports CUDA up to: "
                    f"{'.'.join(str(part) for part in driver_cuda) if driver_cuda else 'unknown'}")

    for item in compat:
        if item["status"] == "warning":
            status = report.worst_status(status, report.WARNING)
            evidence.append(f"compatibility: {item['message']}")
            suggestions.append(f"update the NVIDIA driver for {item['name']} "
                               f"(needs CUDA >= {item['required_cuda']}, currently supports {item['driver_cuda']})")

    if facts["wsl"]["is_wsl"]:
        wsl_lib = Path("/usr/lib/wsl/lib/libcuda.so.1")
        evidence.append(f"WSL2: /usr/lib/wsl/lib/libcuda.so.1 present = {wsl_lib.exists()} "
                        "(CUDA on WSL2 comes from the Windows host driver)")
        wrong_driver = _linux_driver_libs()
        if wrong_driver:
            status = report.worst_status(status, report.WARNING)
            evidence.append(f"Linux NVIDIA driver libraries found in standard paths: {', '.join(wrong_driver)}")
            suggestions.append("if you installed a Linux NVIDIA driver inside WSL2, remove it: "
                               "it conflicts with the Windows driver passthrough and usually breaks CUDA")

    evidence.append("note: the core mujoco package never needs CUDA; CUDA only matters for MJX")

    message = (f"{len(gpus)} NVIDIA GPU(s) detected; CUDA is not required by core mujoco"
               if status == report.PASS else
               "NVIDIA GPU(s) detected, but the driver/compute-capability combination needs attention")
    details = {
        "gpus": gpus,
        "driver_cuda": ".".join(str(part) for part in driver_cuda) if driver_cuda else None,
        "compatibility": compat,
        "mjx_packages": mjx,
        "is_wsl": facts["wsl"]["is_wsl"],
    }
    return report.result(CHECK_ID, TITLE, LEVEL, status, message,
                         details=details, evidence=evidence, suggestions=suggestions)


if __name__ == "__main__":
    report.run_check_cli(CHECK_ID, TITLE, LEVEL, run)
