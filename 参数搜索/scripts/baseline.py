"""搜索线基线（v2）——全部搜索线脚本的唯一参数来源。

2026-09-25 用户确定，对齐 Zhou 等 2026（Signal Processing 245）的仿真口径：

    信号侧：B = 10 MHz、T_p = 25.6 μs、码长 L = B·T_p = 256、fs = B ⇒ 采样间隔 0.1 μs
    干扰侧：T_s = 6.4 μs（64 点/周期）、T_L = 1.5 μs（15 点）、占空比 d = 23.4%
    窗口起点：t_0 = 2 μs（**用户指定**；论文式(15)的原位窗从 0 开始）
    约束：    a_min = a_max·10⁻⁴ ⇒ 零延迟干扰电平约束 −80 dB
    模型：    保持**零延迟原位**（y_j = x⊙j），不含论文的 T_L 延迟转发（决策④）
    误差范围：±0.5 μs（论文 Fig.12 的扫描区间）

与 v1 的差别（v1 = 自建 800 点体系：B = fs = 20 MHz、T_p = 40 μs、码长 800、
t_0 = 2.5 μs、T_L = 2.5 μs、T_s = 10 μs、约束 −30 dB）——本项目**不继承**论文的
ISAC 部分（通信波形 s、QPSK、PAPR = 1.5 的通信约束、多普勒区间），初始化仍用
`random_phase`。

派生量与自洽性（全部整除，无取整误差）：
    1 采样点 = 0.1 μs；每片 15 点；每周期 64 点；码内 4 个采样片（K = 4，与论文一致）。

用法::

    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from baseline import N, T0_US, params_for, summary
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from isrj_joint_design.params import GlobalParams  # noqa: E402

# ---- 信号侧（论文 §4.1）----
B_HZ = 10e6                     # 带宽
FS_HZ = 10e6                    # 复采样率 = B
T_P_S = 25.6e-6                 # 脉宽
N = 256                         # 码长 L = B·T_p = 256
F_C_HZ = 1e9                    # 载频（沿用复现线；论文未强调）

# ---- 干扰侧（论文 §4.1 + 用户指定的窗口起点）----
T0_US = 2.0                     # 转发起始时间（窗口起点）[μs]，用户定
TAU_US = 1.5                    # 截取长度 T_L [μs]
TS_US = 6.4                     # 采样重复周期 T_s [μs]
PEAK_INTERF_DB = -80.0          # 零延迟干扰约束 [dB]（论文 a_min = a_max·1e-4）

# ---- 其他 ----
SEED = 0
INIT = "random_phase"
SAMPLE_US = 1e6 / FS_HZ         # 1 个采样点 = 0.1 μs
ERR_RANGE_US = 0.5              # 论文 Fig.12 的误差区间 ±0.5 μs


def params_for(t0_us: float | None = None, tau_us: float | None = None,
               ts_us: float | None = None, **overrides) -> GlobalParams:
    """按基线构造参数；三个干扰参数可单独覆盖（用于扫描误差）。"""
    kw = dict(N=N, T_p=T_P_S, B=B_HZ, f_c=F_C_HZ, fs=FS_HZ,
              tau=(TAU_US if tau_us is None else tau_us) * 1e-6,
              T_s=(TS_US if ts_us is None else ts_us) * 1e-6,
              tau_offset=(T0_US if t0_us is None else t0_us) * 1e-6,
              peak_interf_db=PEAK_INTERF_DB, seed=SEED, init=INIT)
    kw.update(overrides)
    return GlobalParams(**kw)


def derived() -> dict:
    """派生量（点数），用于打印与文档核对。"""
    return {
        "sample_us": SAMPLE_US,
        "points_per_period": int(round(TS_US / SAMPLE_US)),      # 64
        "points_per_slice": int(round(TAU_US / SAMPLE_US)),      # 15
        "points_offset": int(round(T0_US / SAMPLE_US)),          # 20
        "slices_in_code": N / (TS_US / SAMPLE_US),               # 4.0
        "duty": TAU_US / TS_US,                                  # 0.234
        "b_max": float(N),
        "b_min": float(N * 10 ** (PEAK_INTERF_DB / 20)),
    }


def summary() -> str:
    d = derived()
    return "\n".join([
        "搜索线基线 v2（对齐 Zhou 2026）",
        f"  B = {B_HZ / 1e6:g} MHz、T_p = {T_P_S * 1e6:g} μs、码长 L = {N}、"
        f"fs = {FS_HZ / 1e6:g} MHz（1 点 = {SAMPLE_US:g} μs）",
        f"  t_0 = {T0_US:g} μs（{d['points_offset']} 点）、"
        f"T_L = {TAU_US:g} μs（{d['points_per_slice']} 点）、"
        f"T_s = {TS_US:g} μs（{d['points_per_period']} 点）",
        f"  占空比 d = {d['duty']:.3f}、码内采样片数 K = {d['slices_in_code']:g}",
        f"  约束 b_max = {d['b_max']:g}、b_min = {d['b_min']:.5g}"
        f"（{PEAK_INTERF_DB:g} dB）",
        f"  模型：零延迟原位 y_j = x⊙j；初始化 {INIT}、seed {SEED}",
    ])


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(summary())
