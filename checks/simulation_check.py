"""Level 2：物理引擎。

用一个内置的最小模型（地板 + 自由落体小球）证明 MuJoCo 的编译、积分与碰撞管线真实可用：
MjModel / MjData 能创建，mj_step 能推进，小球最终落在地板上并产生接触。
"""

from __future__ import annotations

import time

from utils import report

CHECK_ID = "simulation"
TITLE = "Simulation"
LEVEL = 2

STEPS = 1500                    # timestep=0.002s -> 3 秒仿真，足够小球落地并静止
BALL_RADIUS = 0.05
REST_Z_RANGE = (0.02, 0.09)     # 静止高度应接近球半径（允许少量接触穿透）

DEMO_XML = f"""
<mujoco>
  <option timestep="0.002"/>
  <worldbody>
    <geom name="floor" type="plane" size="1 1 0.1"/>
    <body name="ball" pos="0 0 1.0">
      <freejoint/>
      <geom name="ball" type="sphere" size="{BALL_RADIUS}" mass="0.1"/>
    </body>
  </worldbody>
</mujoco>
"""


def run() -> report.CheckResult:
    try:
        import mujoco
        import numpy as np
    except Exception as exc:
        return report.result(CHECK_ID, TITLE, LEVEL, report.SKIPPED,
                             f"mujoco not importable: {exc}",
                             suggestions=["fix the 'MuJoCo import' check first"])

    try:
        model = mujoco.MjModel.from_xml_string(DEMO_XML)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
    except Exception as exc:
        return report.result(CHECK_ID, TITLE, LEVEL, report.FAIL,
                             f"could not build the demo model: {type(exc).__name__}: {exc}",
                             evidence=[f"{type(exc).__name__}: {exc}"],
                             suggestions=["the MuJoCo binary itself looks broken; "
                                          "reinstall: pip install --force-reinstall mujoco"])

    z_start = float(data.qpos[2])
    z_min = z_start
    start = time.perf_counter()
    for step in range(STEPS):
        mujoco.mj_step(model, data)
        if not np.isfinite(np.asarray(data.qpos)).all():
            return report.result(CHECK_ID, TITLE, LEVEL, report.FAIL,
                                 f"simulation produced a non-finite state at step {step}",
                                 evidence=[f"qpos: {data.qpos}"],
                                 suggestions=["the physics engine is misbehaving; "
                                              "reinstall: pip install --force-reinstall mujoco"])
        z_min = min(z_min, float(data.qpos[2]))
    elapsed = max(time.perf_counter() - start, 1e-9)

    z_end = float(data.qpos[2])
    ncon = int(data.ncon)
    evidence = [
        f"ball z: {z_start:.3f} -> {z_end:.3f} (radius {BALL_RADIUS})",
        f"lowest z seen: {z_min:.3f}",
        f"contacts at rest: {ncon}",
        f"{STEPS} steps in {elapsed:.3f} s ({STEPS / elapsed:.0f} steps/s)",
    ]
    details = {
        "z_start": round(z_start, 4),
        "z_end": round(z_end, 4),
        "z_min": round(z_min, 4),
        "contacts": ncon,
        "steps": STEPS,
        "steps_per_second": round(STEPS / elapsed, 1),
    }

    if not (REST_Z_RANGE[0] <= z_end <= REST_Z_RANGE[1]):
        return report.result(CHECK_ID, TITLE, LEVEL, report.FAIL,
                             f"the ball did not settle on the floor (final z={z_end:.3f}, expected ~{BALL_RADIUS})",
                             details=details, evidence=evidence,
                             suggestions=["the integrator or collision pipeline looks broken; "
                                          "reinstall: pip install --force-reinstall mujoco"])
    if ncon < 1:
        return report.result(CHECK_ID, TITLE, LEVEL, report.FAIL,
                             "no contact was detected while the ball rests on the floor",
                             details=details, evidence=evidence,
                             suggestions=["the collision pipeline looks broken; "
                                          "reinstall: pip install --force-reinstall mujoco"])

    return report.result(CHECK_ID, TITLE, LEVEL, report.PASS,
                         "the physics engine stepped the model and the ball settled on the floor",
                         details=details, evidence=evidence)


if __name__ == "__main__":
    report.run_check_cli(CHECK_ID, TITLE, LEVEL, run)
