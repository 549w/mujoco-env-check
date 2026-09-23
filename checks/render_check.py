"""Level 3：渲染。

渲染是"平台差异最大"的一层：headless 服务器、SSH 会话、容器、WSL 里不可用属于常态，
因此任何失败都记为 WARNING，并且只给出"该平台下最可能的原因"。
"""

from __future__ import annotations

from utils import platform_info, report

CHECK_ID = "render"
TITLE = "Rendering"
LEVEL = 3

WIDTH, HEIGHT = 320, 240
MIN_STD = 1.0                 # 图像标准差下限，低于此值视为"疑似空白"
MIN_NON_BG_FRACTION = 0.01    # 与背景色不同的像素占比下限

DEMO_XML = """
<mujoco>
  <worldbody>
    <geom name="floor" type="plane" size="1 1 0.1"/>
    <body name="ball" pos="0 0 0.3">
      <geom name="ball" type="sphere" size="0.2" rgba="0.9 0.3 0.1 1"/>
    </body>
    <camera name="view" pos="1.6 -1.6 1.2" xyaxes="0.707 0.707 0 -0.35 0.35 0.87"/>
  </worldbody>
</mujoco>
"""


def _headless_note(facts: dict):
    """在 headless / 无图形会话的环境里，渲染失败属于预期，需要在报告里说明。"""
    wsl = facts["wsl"]
    if "ssh" in facts["headless_hints"] or "no-display" in facts["headless_hints"]:
        return ("this looks like a headless/remote session, where rendering is often unavailable; "
                "that is expected and does not mean the MuJoCo install is broken")
    if wsl.get("is_wsl") and not wsl.get("wslg"):
        return ("no WSLg detected: plain WSL2 sessions usually cannot render; "
                "that is expected and does not mean the MuJoCo install is broken")
    return None


def _troubleshooting(facts: dict, gl_raw, backend_failed: bool = False) -> list:
    """按平台给出最可能的下一步。"""
    os_name = facts["os"]
    tips = []
    if backend_failed:
        tips.append("the selected GL backend failed to load at import time, so mujoco.Renderer "
                    "does not exist; the libraries for the chosen backend are probably missing")

    if os_name == "macos":
        tips.append("macOS: keep MUJOCO_GL unset (uses 'cgl', which renders offscreen without a window server) "
                    "or use 'glfw' inside a desktop session")
        if gl_raw and gl_raw.strip().lower() in ("egl", "osmesa"):
            tips.append(f"MUJOCO_GL={gl_raw} does not exist on macOS and will break rendering")
        if "ssh" in facts["headless_hints"]:
            tips.append("SSH session detected: 'glfw' needs a logged-in GUI session; CGL usually still works offscreen")
    elif os_name == "linux":
        if facts["wsl"].get("is_wsl"):
            tips.append("WSL2: GLFW works through WSLg; for headless use MUJOCO_GL=egl (libraries live in "
                        "/usr/lib/wsl/lib) or MUJOCO_GL=osmesa after 'apt install libosmesa6'")
            if not facts["wsl"].get("wslg"):
                tips.append("no WSLg detected: expect rendering to be unavailable over plain SSH into WSL")
        else:
            tips.append("Linux: try MUJOCO_GL=egl (install libegl1 and a working GPU driver), "
                        "MUJOCO_GL=osmesa (apt install libosmesa6) for pure software rendering, "
                        "or 'glfw' inside a running X/Wayland session")
            if "no-display" in facts["headless_hints"]:
                tips.append("no DISPLAY/WAYLAND_DISPLAY: 'glfw' cannot open a window here; "
                            "use egl/osmesa for headless rendering")
            tips.append("if the GPU driver is broken, LIBGL_ALWAYS_SOFTWARE=1 forces software OpenGL")
    elif os_name == "windows":
        tips.append("Windows: keep MUJOCO_GL unset (GLFW/WGL) and run from a normal desktop session")

    valid = ", ".join(platform_info.VALID_MUJOCO_GL.get(os_name, []))
    tips.append(f"MUJOCO_GL is currently {'unset' if not gl_raw else gl_raw}; valid values on {os_name}: {valid}")
    return tips


def _image_stats(np, image) -> dict:
    """统计图像是否"有内容"：以出现最多的颜色为背景，看非背景像素占比与整体标准差。"""
    arr = np.asarray(image)
    flat = arr.reshape(-1, arr.shape[-1]) if arr.ndim == 3 else arr.reshape(-1, 1)
    colors, counts = np.unique(flat, axis=0, return_counts=True)
    modal_fraction = float(counts.max()) / float(flat.shape[0])
    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "std": float(flat.std()),
        "unique_colors": int(colors.shape[0]),
        "non_background_fraction": 1.0 - modal_fraction,
    }


def run() -> report.CheckResult:
    facts = platform_info.gather_platform_facts()
    gl_raw = facts["mujoco_gl"]
    gl_kind, gl_msg = platform_info.classify_mujoco_gl(facts["os"], gl_raw)
    details = {
        "mujoco_gl": gl_raw,
        "mujoco_gl_kind": gl_kind,
        "default_backend": platform_info.DEFAULT_MUJOCO_GL.get(facts["os"]),
        "headless_hints": facts["headless_hints"],
        "wsl": facts["wsl"],
    }

    if gl_kind == "off":
        return report.result(CHECK_ID, TITLE, LEVEL, report.SKIPPED, gl_msg,
                             details=details, evidence=[gl_msg],
                             suggestions=["unset MUJOCO_GL to re-enable render verification"])

    try:
        import mujoco
        import numpy as np
    except Exception as exc:
        return report.result(CHECK_ID, TITLE, LEVEL, report.SKIPPED,
                             f"mujoco not importable: {exc}",
                             details=details, evidence=[f"{type(exc).__name__}: {exc}"],
                             suggestions=["fix the 'MuJoCo import' check first"])

    if not hasattr(mujoco, "Renderer"):
        # Linux 上 MUJOCO_GL=osmesa 但缺少 libOSMesa 时，mujoco 会静默吞掉 ImportError
        return report.result(CHECK_ID, TITLE, LEVEL, report.WARNING,
                             "mujoco imported, but mujoco.Renderer is missing: the GL backend failed to load "
                             "and MuJoCo swallowed that error",
                             details=details, evidence=[gl_msg],
                             suggestions=_troubleshooting(facts, gl_raw, backend_failed=True))

    renderer = None
    try:
        model = mujoco.MjModel.from_xml_string(DEMO_XML)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "view")
        renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)
        renderer.update_scene(data, camera=camera_id if camera_id >= 0 else -1)
        image = renderer.render()
    except Exception as exc:
        evidence = [f"{type(exc).__name__}: {exc}", gl_msg]
        note = _headless_note(facts)
        if note:
            evidence.append(note)
        return report.result(CHECK_ID, TITLE, LEVEL, report.WARNING,
                             f"renderer failed: {type(exc).__name__}: {exc}",
                             details=details, evidence=evidence,
                             suggestions=_troubleshooting(facts, gl_raw))
    finally:
        if renderer is not None:
            try:
                renderer.close()
            except Exception:
                pass

    stats = _image_stats(np, image)
    evidence = [
        gl_msg,
        f"image: {stats['shape']} {stats['dtype']}",
        (f"std: {stats['std']:.2f}, non-background pixels: "
         f"{stats['non_background_fraction'] * 100:.1f}%, distinct colors: {stats['unique_colors']}"),
    ]
    details["image"] = stats

    if stats["std"] < MIN_STD or stats["non_background_fraction"] < MIN_NON_BG_FRACTION:
        return report.result(CHECK_ID, TITLE, LEVEL, report.WARNING,
                             "the renderer produced an image, but it looks blank",
                             details=details, evidence=evidence,
                             suggestions=_troubleshooting(facts, gl_raw))

    return report.result(CHECK_ID, TITLE, LEVEL, report.PASS,
                         f"rendered a {WIDTH}x{HEIGHT} frame from an offscreen camera",
                         details=details, evidence=evidence)


if __name__ == "__main__":
    report.run_check_cli(CHECK_ID, TITLE, LEVEL, run)
