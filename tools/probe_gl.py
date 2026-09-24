"""Probe which GL rasterizer actually draws MuJoCo frames: CPU or GPU.

Rendering can pass the main check while silently falling back to Mesa's
software rasterizer (llvmpipe) - common on WSL2. Two signals separate the
cases: the GL_RENDERER string of the live context, and how frame time
scales with resolution (a software rasterizer scales with pixel count,
a real GPU scales much more slowly).

Standalone helper: does not touch the environment and is not part of the
main report. Run from the repo root:

    python3 tools/probe_gl.py [--frames N]
"""

from __future__ import annotations

import argparse
import os
import time

SIZES = (64, 512, 2048)
WARMUP = 5

# offwidth/offheight must cover the largest probe size: mujoco.Renderer refuses
# sizes above the offscreen framebuffer (default 640x480) at construction time.
DEMO_XML = """
<mujoco>
  <visual>
    <global offwidth="2048" offheight="2048"/>
  </visual>
  <worldbody>
    <geom name="floor" type="plane" size="1 1 0.1"/>
    <body name="ball" pos="0 0 0.3">
      <geom name="ball" type="sphere" size="0.2" rgba="0.9 0.3 0.1 1"/>
    </body>
    <camera name="view" pos="1.6 -1.6 1.2" xyaxes="0.707 0.707 0 -0.35 0.35 0.87"/>
  </worldbody>
</mujoco>
"""


def _gl_strings():
    """GL_RENDERER / GL_VERSION of the current context; (None, None) if unavailable."""
    try:
        from OpenGL import GL

        def read(name):
            value = GL.glGetString(name)
            return value.decode("utf-8", "replace") if isinstance(value, bytes) else str(value)

        return read(GL.GL_RENDERER), read(GL.GL_VERSION)
    except Exception:
        return None, None


def _timed_render(mujoco, model, data, camera_id, size, frames):
    """Render `frames` timed frames at size x size; returns (ms_per_frame, gl_strings)."""
    renderer = mujoco.Renderer(model, height=size, width=size)
    try:
        renderer.update_scene(data, camera=camera_id)
        for _ in range(WARMUP):
            renderer.render()
        start = time.perf_counter()
        for _ in range(frames):
            renderer.render()
        ms = (time.perf_counter() - start) / frames * 1000.0
        return ms, _gl_strings()
    finally:
        renderer.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render the same scene at several resolutions and report ms/frame + GL renderer.")
    parser.add_argument("--frames", type=int, default=30,
                        help="timed frames per resolution (default: 30)")
    args = parser.parse_args()
    if args.frames < 1:
        parser.error("--frames must be >= 1")

    gl_env = os.environ.get("MUJOCO_GL", "").strip()
    print(f"MUJOCO_GL: {gl_env or 'unset (platform default)'}")
    if gl_env.lower() == "disable":
        print("FAIL: MUJOCO_GL=disable turns rendering off by design; nothing to probe.")
        print("Unset MUJOCO_GL (or choose a backend) to run this probe.")
        return 1
    if gl_env.lower() in ("egl", "osmesa"):
        os.environ.setdefault("PYOPENGL_PLATFORM", gl_env.lower())

    try:
        import mujoco
    except Exception as exc:
        print(f"FAIL: cannot import mujoco: {type(exc).__name__}: {exc}")
        if isinstance(exc, RuntimeError):
            print("This usually means MUJOCO_GL holds an invalid value for this platform;")
            print("see README, or unset it to use the platform default backend.")
        else:
            print("Install it first: pip install mujoco (into this same interpreter).")
        return 1

    if not hasattr(mujoco, "Renderer"):
        print("FAIL: mujoco.Renderer is missing: the GL backend did not load and MuJoCo")
        print("swallowed the error (e.g. MUJOCO_GL=osmesa without libOSMesa on Linux).")
        print("Run main.py; its Rendering section explains this per platform.")
        return 1

    model = mujoco.MjModel.from_xml_string(DEMO_XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "view")

    try:
        _, (renderer_string, version) = _timed_render(mujoco, model, data, camera_id, SIZES[0], 1)
        if renderer_string:
            print(f"GL renderer: {renderer_string}")
            print(f"GL version : {version}")
        print()
        print(f"{args.frames} timed frames per resolution (+{WARMUP} warmup)")
        print()
        print(f"{'resolution':<14} {'ms/frame':>8}    relative")
        timings = []
        for size in SIZES:
            ms, _ = _timed_render(mujoco, model, data, camera_id, size, args.frames)
            timings.append(ms)
            relative = ms / timings[0] if timings[0] > 0 else 0.0
            print(f"{f'{size}x{size}':<14} {ms:8.2f}    x{relative:.1f}")
    except Exception as exc:
        print(f"FAIL: renderer failed: {type(exc).__name__}: {exc}")
        print("Run main.py; its Rendering section explains the failure per platform,")
        print("and the README Troubleshooting section covers the common cases.")
        return 1

    print()
    print("A software rasterizer's frame time roughly follows the pixel count; a real GPU's")
    print("grows much more slowly (glReadPixels still costs, so it is not flat either).")
    print("Cross-check the 'GL renderer' line above: llvmpipe / softpipe means software.")
    print("The ratio is the signal, not the raw speed.")
    print()
    print("Tip: run with a large --frames while watching Task Manager / nvidia-smi to see")
    print("whether the GPU stays busy throughout rendering or only blips once at startup.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
