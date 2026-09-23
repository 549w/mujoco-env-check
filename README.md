# mujoco-env-check

跨平台 MuJoCo 环境检测工具：`git clone` 之后跑一条命令，得到一份"这台机器的 MuJoCo 环境到底能不能用"的诊断报告。

## Purpose

在不同设备（macOS / Linux / Windows / WSL2）上配置 MuJoCo 时，真正的问题往往不是"能不能装上"，
而是"装完之后到底能不能用、哪一层出了问题"。这个仓库把"人判断 MuJoCo 环境是否配置成功"的经验
固化成了可执行的 verification：

- 按 Level 0-4 分层检查，定位问题发生在哪一层（系统 / 绑定 / 物理引擎 / 渲染 / GPU）
- 每个检查给出可观察证据（evidence）而不是 True/False，例如"小球从 z=1.000 落到 z=0.050 并产生接触"
- 输出终端报告 + `reports/latest.json`，后者可供未来的"环境配置 Agent"直接读取

**这个仓库不是安装器**：它不安装、不修改、不"修复"任何环境，只做检测和报告。

## Usage

### 前置要求

| 角色 | 要求 |
| --- | --- |
| 运行检测工具本身 | Python >= 3.8（`run.sh` / `run.ps1` 会自动寻找解释器） |
| 被检测的 MuJoCo 环境 | Python >= 3.10，且已安装 `mujoco`（`pip install mujoco`） |

工具本身只依赖 Python 标准库；`mujoco` 只在被检测时才会 import（未安装也能正常出报告）。

### 快速开始

```bash
git clone <this-repo-url>
cd mujoco-env-check
./run.sh
```

Windows PowerShell：

```powershell
.\run.ps1
# 如遇执行策略限制：
powershell -ExecutionPolicy Bypass -File .\run.ps1
```

### 常用参数

```bash
./run.sh --only render            # 只跑某一项（可重复：--only mujoco --only gpu），调试时很有用
./run.sh --timeout 180            # 覆盖所有检查的超时秒数（默认 system 10s / mujoco 20s / simulation 60s / render 90s / gpu 15s）
```

单独运行某一项检查（输出为一行 JSON，最方便排查）：

```bash
python3 -m checks.render_check
```

## Verification philosophy

**为什么 `import mujoco` 成功 ≠ 环境完整。**
import 只证明 wheel 能被加载，连 mj_step 能不能推进、Renderer 能不能出图都没证明。
所以检查被分成三层：包可加载（Level 1）→ 物理引擎能跑（Level 2）→ 渲染能用（Level 3）。
一个"能 import 但仿真崩"的环境和一个"能 import 但渲染不可用"的环境，故障点和修复方式完全不同。

**为什么 simulation 和 rendering 必须分开判定。**
仿真（MjModel/MjData/mj_step）是 MuJoCo 的核心能力，坏了就是环境坏了；
渲染是平台差异最大的一层：headless 服务器、SSH 会话、容器、无 WSLg 的 WSL2 里渲染不可用属于常态，
不应该据此判定"环境配置失败"。因此渲染失败只记 WARNING，不影响整体结论。

**为什么 GPU 只是信息项。**
核心 `mujoco` 包永远不需要 CUDA（CUDA 只与 MJX / `mujoco-mjx` 相关）。
GPU 检查用于发现"Blackwell 显卡 + 老驱动"这类隐性不匹配，但任何结果都不影响核心结论。

**为什么有"依赖短路"。**
`import mujoco` 失败时，simulation / rendering 会被标记为 `SKIPPED`（而不是再报一堆错），
让报告的第一屏就指向唯一的根因。

## 检查层级

| Level | 检查 | 失败影响 |
| --- | --- | --- |
| 0 | System：OS / 架构 / Python 版本门禁 / venv·conda / Rosetta / WSL / wheel 覆盖 / `MUJOCO_GL` 合法性 | FAIL（核心） |
| 1 | MuJoCo import：`import mujoco`、包版本 vs 运行时版本、numpy | FAIL（核心） |
| 2 | Simulation：内置最小模型跑 `mj_step`，小球落地并产生接触 | FAIL（核心） |
| 3 | Rendering：`mujoco.Renderer` 出图并检查非空白 | WARNING（永不影响 exit code） |
| 4 | GPU：nvidia-smi / 驱动与算力匹配 / WSL 驱动透传 / MJX 包探测 | 信息项 |

## Supported platforms

| 平台 | 渲染后端（默认 / 可选） | GPU 检查行为 | 注意 |
| --- | --- | --- | --- |
| macOS (Apple Silicon) | `cgl` / `glfw` | Not applicable（无 CUDA） | CGL 离屏渲染在无窗口会话下通常仍可用；**macOS Intel 无官方 wheel**（mujoco >= 3.11） |
| Linux 桌面 | `glfw` / `glx`, `egl`, `osmesa` | nvidia-smi + 驱动/算力匹配 | 无 `DISPLAY` 时 `glfw` 会快速失败，属预期 |
| Linux headless / 容器 | `egl` 或 `osmesa` | 同上 | 渲染失败记 WARNING，不影响核心结论 |
| Windows | `glfw` / `wgl` | nvidia-smi | 建议在正常桌面会话中运行；终端输出为纯 ASCII，避免 GBK 控制台乱码 |
| WSL2（有 WSLg） | `glfw`（经 WSLg）/ `egl`, `osmesa` | nvidia-smi（CUDA 来自 Windows 宿主驱动） | **不要在 WSL 内安装 Linux NVIDIA 驱动**，会破坏驱动透传 |
| WSL2（无 WSLg，纯 SSH） | `egl` 或 `osmesa` | 同上 | 渲染不可用属预期 |

`MUJOCO_GL` 的合法取值按平台区分（非法值会让 `import mujoco` 直接失败）：

| 平台 | 合法值 |
| --- | --- |
| macOS | `cgl`, `glfw` |
| Linux | `egl`, `glfw`, `glx`, `osmesa` |
| Windows | `glfw`, `wgl` |

所有平台都接受 `disable`（显式关闭渲染，检测结果记为 SKIPPED，而不是故障）。

## Output format

终端报告（未安装 mujoco 时的真实输出，已截断）：

```
================================================
MuJoCo Environment Report
================================================

System
------
OS: Darwin 24.6.0
Arch: arm64
Python: 3.13.9
Context: native macOS

Checks
------

[PASS] System information (Level 0)
      Darwin 24.6.0, arm64, Python 3.13.9 (venv)
      - Python 3.13.9 satisfies mujoco's requirement (>= 3.10)
      - official mujoco wheels exist for macOS arm64
      - MUJOCO_GL not set; the platform default backend is 'cgl'

[FAIL] MuJoCo import (Level 1)
      mujoco package not found: No module named 'mujoco'
      - ImportError: No module named 'mujoco'
      Suggest: pip install mujoco

[SKIPPED] Simulation (Level 2)
      mujoco is not importable (see the 'MuJoCo import' check)

[SKIPPED] Rendering (Level 3)
      mujoco is not importable (see the 'MuJoCo import' check)

Summary
-------

Core MuJoCo environment: FAIL
Rendering: SKIPPED
GPU: SKIPPED
================================================
```

环境正常时（`:> .venv` + `pip install mujoco` 后的真实输出，已截断）：

```
[PASS] MuJoCo import (Level 1)
      mujoco 3.13.0 imported successfully
      - package version: 3.13.0
      - runtime version (mj_versionString): 3.13.0
      - header version (mjVERSION_HEADER): 3013000
      - numpy version: 2.5.3

[PASS] Simulation (Level 2)
      the physics engine stepped the model and the ball settled on the floor
      - ball z: 1.000 -> 0.050 (radius 0.05)
      - lowest z seen: 0.021
      - contacts at rest: 1
      - 1500 steps in 0.004 s (419111 steps/s)

[PASS] Rendering (Level 3)
      rendered a 320x240 frame from an offscreen camera
      - MUJOCO_GL not set; the platform default backend is 'cgl'
      - image: [240, 320, 3] uint8
      - std: 19.61, non-background pixels: 40.9%, distinct colors: 276

[SKIPPED] GPU acceleration (Level 4)
      Not applicable: macOS has no CUDA path (Apple GPU; graphics is handled by CGL)
```

同时每次运行都会写出 `reports/latest.json`（即使全部检查失败也会写），结构：

```json
{
  "schema_version": 1,
  "timestamp": "2026-01-01T00:00:00+00:00",
  "system": { "os": "...", "arch": "...", "python": "...", "context": "...", "...": "..." },
  "checks": {
    "mujoco": {
      "check_id": "mujoco",
      "title": "MuJoCo import",
      "level": 1,
      "status": "FAIL",
      "message": "mujoco package not found: No module named 'mujoco'",
      "details": {},
      "evidence": ["ImportError: No module named 'mujoco'"],
      "suggestions": ["pip install mujoco"],
      "duration_ms": 120
    }
  },
  "summary": { "core": "FAIL", "rendering": "SKIPPED", "gpu": "SKIPPED", "exit_code": 1 }
}
```

状态取值：`PASS` / `WARNING`（非核心问题）/ `FAIL`（核心问题）/ `SKIPPED`（不适用或前置依赖缺失）/
`ERROR`（该检查自身崩溃或超时）。

## Exit codes

| 退出码 | 含义 |
| --- | --- |
| 0 | 核心环境可用（system / mujoco / simulation 均未 FAIL） |
| 1 | 核心环境不可用 |
| 2 | 检测工具自身无法运行（例如找不到 Python） |

渲染与 GPU 的 WARNING / SKIPPED **不会**改变退出码。

## Adding a check

1. 新建 `checks/<name>_check.py`，实现 `run() -> CheckResult`：

```python
from utils import report

CHECK_ID = "mycheck"
TITLE = "My check"
LEVEL = 5

def run() -> report.CheckResult:
    return report.result(CHECK_ID, TITLE, LEVEL, report.PASS, "everything is fine")

if __name__ == "__main__":
    report.run_check_cli(CHECK_ID, TITLE, LEVEL, run)
```

2. 在 `main.py` 的 `CHECKS` 注册表里加一行（id / module / title / level / timeout）。
3. 需要的话在 `utils/platform_info.py` 里加平台知识表（保持"判定是纯函数、读取是薄封装"的风格，便于用合成输入验证）。

## Troubleshooting

**`import mujoco` 失败（FAIL: mujoco package not found）**
- 先确认解释器：报告 System 区块里的 `Python` 行会显示解释器路径与 venv/conda 归属，`pip install mujoco` 必须装进同一个环境。
- Python 必须 >= 3.10（更老的版本没有对应 wheel）。
- wheel 覆盖范围有限：macOS 仅 arm64（Intel Mac 自 mujoco 3.11 起没有 wheel）、Windows 仅 amd64、Linux x86_64 / aarch64。
- macOS 报 Rosetta 相关错误：解释器是 x86_64 跑在 Apple Silicon 上，换 arm64 的 Python。

**`import mujoco` 抛 RuntimeError（多为 MUJOCO_GL 配置问题）**
- `MUJOCO_GL` 在 import 阶段就会校验：非法值（例如在 macOS 上设 `osmesa`）会让 import 直接失败。
- 对照上文"Supported platforms"里的合法值表，或直接 unset `MUJOCO_GL` 用平台默认值。

**Rendering 是 WARNING**
- 先看报告里的 `Context:` 行：SSH 会话 / 无 DISPLAY / 无 WSLg 的 WSL2 上渲染不可用通常是预期行为。
- Linux headless：`MUJOCO_GL=egl`（需要 libegl1 和可用驱动）或 `MUJOCO_GL=osmesa`（`apt install libosmesa6`）。
  注意 Linux 上 `MUJOCO_GL=osmesa` 缺库时 MuJoCo 会静默吞掉错误、`mujoco.Renderer` 直接消失——报告会明确提示这种情形。
- 驱动异常时可用 `LIBGL_ALWAYS_SOFTWARE=1` 强制软件渲染。
- macOS：保持 `MUJOCO_GL` 不设置（默认 `cgl`，离屏可用）；`glfw` 需要桌面会话。

**WSL2 相关**
- 有 WSLg：渲染可走 `glfw`（经 WSLg 的 X/Wayland），或 `egl`。
- 无 WSLg（纯 SSH）：用 `osmesa` / `egl`，渲染不可用属预期。
- CUDA 来自 Windows 宿主驱动透传（`/usr/lib/wsl/lib/libcuda.so.1`），**不要在 WSL 里安装 Linux NVIDIA 驱动**；报告会在检测到冲突时提示。

**Blackwell（RTX 50 系 / B100/B200）相关**
- 报告会对比 GPU 算力等级与驱动自带 CUDA 版本：Blackwell（算力 10.x / 12.x）需要 CUDA 12.8+，
  老驱动会得到一条"请升级 NVIDIA 驱动"的 WARNING。
- 再次强调：这只影响 MJX（`jax` / `mujoco_mjx`），核心 mujoco 不需要 CUDA。

**Windows**
- 若 `.\run.ps1` 被策略拦截：`powershell -ExecutionPolicy Bypass -File .\run.ps1`。
- 建议在普通桌面会话运行（服务/SSH 上下文里没有窗口站，`glfw` 渲染会失败）。

## 已知限制

- 平台判定基于启发式（`/proc/version`、环境变量、文件存在性），覆盖常见发行版与 WSL 组合，但不保证穷尽特例。
- WSL / Blackwell 等分支的判定逻辑在本机（macOS）以合成输入验证；真实的端到端行为需要在对应平台上运行确认。
- 不做性能基准，`steps/s` 只是信息（随机负载下波动大）。
- 不安装、不修复、不修改任何环境配置。
