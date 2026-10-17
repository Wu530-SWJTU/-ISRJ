# -*- coding: utf-8 -*-
"""对抗场景第 2 步（2026-10-01 重做）：随机相位波形 + 匹配滤波被 ISRJ
直发转发干扰，假目标串 + 噪声，画幅度-距离图。

干扰模型（重复转发 R=4，评估侧）：
    信号级   y_jam = Σ_{r=0}^{3} shift_{τ_j + r·T_L}(x ⊙ j)，τ_j = 2 μs
    矩阵链   Y_jam = S_{τ_j}·(I + S_{T_L} + S_{2T_L} + S_{3T_L})·diag(j)·x
             脉压输出端 H_j = H·S_{τ_j}·(I+S_{T_L}+S_{2T_L}+S_{3T_L})·diag(j)
    假目标   τ_j + r·T_L（r = 0..3）= 真目标后 2/3.5/5/6.5 μs（间距 225 m）：
             每份转发同片完全对齐 ⇒ 4 片 × 15 点 × a_j 满能量，各 −12.6 dB + J/S；
             随机相位峰宽仅 1 点 ⇒ 相邻假目标不重叠。

其余口径同第 1 步：随机相位波形（未优化）、匹配滤波、SNR=10 dB、J/S=10 dB。
不做 CFAR 检测（2026-10-01 用户定）。

用法：
    .venv/Scripts/python.exe 参数搜索/scripts/run_scene_jamming.py [--jsr 10]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import baseline  # noqa: E402
from isrj_joint_design.params import generate_isrj_sampling_function  # noqa: E402

OUTDIR = ROOT / "参数搜索" / "结果" / "对抗场景"
TAU_T_US = 30.0            # 真目标时延（μs）
R_T_KM = 30.0              # 真目标距离（km）
KM_PER_US = 0.15
WIN_KM = (-4.0, 4.0)       # 图窗：真目标 ±4 km
SNR_DB = 10.0              # 匹配滤波前 SNR（2026-10-01 定）
SEED = 2                   # 噪声种子（可复现；seed=1 在 29.8 km 处有 1 个噪声虚警）
JSR_DB = 10.0              # 干信比（单份转发幅度相对目标回波；2026-10-01 用户定）
TAU_J_US = 2.0             # 转发时延 τ_j（μs）
# 经典间歇采样重复转发（2026-10-01 用户定 Q3）：
#   切片 s 在 [t_0+s·T_s, t_0+s·T_s+T_L) 采样，采样结束立即重复转发 R 次，
#   第 r 次占用 [t_0+s·T_s+r·T_L, t_0+s·T_s+(r+1)·T_L)
#   ⇒ y_jam = Σ_{r=1}^{R} shift_{r·T_L}(x⊙j)，假目标在 +r·T_L（只能为正：
#     转发必须先收到再发出）
#   物理约束 (1+R)·T_L ≤ T_s：v2 基线 T_L=1.5、T_s=6.4 ⇒ R ≤ 3
T_L_US = baseline.TAU_US   # 切片长度（唯一来源：baseline.py）
R_RETX = int(baseline.TS_US / baseline.TAU_US) - 1   # 占空比约束 (1+R)·T_L ≤ T_s
OFFSETS_US = [r * T_L_US for r in range(1, R_RETX + 1)]   # +1.5, +3.0, +4.5 μs
TEXT_PRIMARY, TEXT_SECONDARY = "#0b0b0b", "#52514e"
SURFACE = "#fcfcfb"


def scene_output(x: np.ndarray, j: np.ndarray, tau_t_samples: int) -> tuple:
    """接收 = 真目标 + R 份重复转发干扰 + 白噪声；匹配滤波输出与距离轴（km）。

    转发均为正时延（边采样边转发：先收到再发出），接收序列长度按最大偏移扩展。
    """
    n = len(x)
    a_j = 10 ** (JSR_DB / 20)
    offs = [int(round(u / baseline.SAMPLE_US)) for u in OFFSETS_US]
    lo, hi = min(0, min(offs)), max(0, max(offs))

    r = np.zeros(hi - lo + n, dtype=complex)
    r[-lo:-lo + n] += x                      # 真目标
    for off in offs:
        start = -lo + off
        r[start:start + n] += a_j * (x * j)  # 各份转发（含负时延）

    sigma = 10 ** (-SNR_DB / 20)
    rng = np.random.default_rng(SEED)
    r += (rng.standard_normal(len(r)) + 1j * rng.standard_normal(len(r))) \
        * (sigma / np.sqrt(2))

    y = np.convolve(r, np.conj(x[::-1]), mode="full")
    peak_idx = -lo + n - 1
    ref = (np.abs(x) ** 2).sum()
    r_km = R_T_KM + (np.arange(len(y)) - peak_idx) * baseline.SAMPLE_US * KM_PER_US
    y_db = 20 * np.log10(np.abs(y) / ref + 1e-30)
    return r_km, y_db


def main() -> None:
    global JSR_DB
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jsr", type=float, default=JSR_DB,
                    help="干信比 dB（默认 15 = 正式口径）")
    JSR_DB = ap.parse_args().jsr

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    OUTDIR.mkdir(parents=True, exist_ok=True)

    x = baseline.random_phase_waveform()   # 未优化随机相位波形（2026-10-01 用户改）
    j = generate_isrj_sampling_function(baseline.params_for())
    tau_t_samples = int(round(TAU_T_US / baseline.SAMPLE_US))
    r_km, y_db = scene_output(x, j, tau_t_samples)

    m = (r_km >= R_T_KM + WIN_KM[0]) & (r_km <= R_T_KM + WIN_KM[1])
    rk = r_km[m]
    yk_db = y_db[m]

    # ---- 理论假目标（随机相位波形，经典重复转发 R 次）----
    # 每份转发都同片完全对齐 ⇒ 4 片 × 15 点 × a_j 满能量，各 −12.6 dB + J/S；
    # 位置 = +r·T_L（只能为正，转发必须先收到再发出）。
    e_slice = (np.abs(x * j) ** 2).sum()               # 4 片能量 ≈ 60
    e_total = (np.abs(x) ** 2).sum()                   # 256
    amp_db = 20 * np.log10(e_slice / e_total) + JSR_DB
    false_targets = [(k, R_T_KM + u * KM_PER_US, amp_db)
                     for k, u in enumerate(OFFSETS_US)]
    print(f"理论假目标（随机相位波形，经典重复转发 R={R_RETX} 次）：")
    for k, pos, amp in false_targets:
        print(f"   FT{k + 1} @ {pos:7.3f} km（真目标后 "
              f"{OFFSETS_US[k]:g} μs = {OFFSETS_US[k] * KM_PER_US * 1000:.0f} m）"
              f"   {amp:+7.2f} dB")
    print()

    # ---- 实测假目标电平（各理论位置 ±50 m 内取最大）----
    print("实测：")
    for k, pos, amp in false_targets:
        near = np.abs(rk - pos) <= 0.05
        meas = float(yk_db[near].max()) if near.any() else float("nan")
        print(f"   FT{k + 1} @ {pos:7.3f} km   理论 {amp:+6.2f} dB   实测 {meas:+6.2f} dB")
    i_true = int(np.argmin(np.abs(rk - R_T_KM)))
    print(f"   真目标 @ {R_T_KM:7.3f} km   实测 {yk_db[i_true]:+6.2f} dB "
          f"（无干扰时为 0 dB）")

    # ---- 绘图 ----
    fig, ax = plt.subplots(figsize=(11, 5.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.plot(rk, yk_db, color="#2a6f97", lw=0.9, zorder=2,
            label="matched filter output")
    ax.axvline(R_T_KM, color="#e4572e", lw=0.9, ls="--", zorder=3,
               label="true target (30 km)")
    for rr, p, _ in false_targets:
        near = np.abs(rk - p) <= 0.05
        meas = float(yk_db[near].max()) if near.any() else float("nan")
        ax.axvline(p, color="#c9a227", lw=0.8, ls="--", alpha=0.85, zorder=3)
        ax.scatter([p], [meas], color="#c9a227", s=26, zorder=4,
                   edgecolor="white", linewidth=0.5)
        ax.annotate(f"FT{rr + 1}  {meas:+.1f} dB", (p, 2.6), ha="center",
                    fontsize=8.5, color="#8a6d1a",
                    bbox=dict(boxstyle="round,pad=0.15", fc="#fdf6e3",
                              ec="#c9a227", lw=0.6))
    ax.axhline(0.0, color=TEXT_SECONDARY, lw=0.5, ls=":")
    ax.set_xlim(R_T_KM + WIN_KM[0], R_T_KM + WIN_KM[1])
    ax.set_ylim(-60, 8)
    ax.set_xlabel("Range (km)", color=TEXT_SECONDARY)
    ax.set_ylabel("Normalized level (dB)", color=TEXT_SECONDARY)
    ax.set_title("Step 2 — random-phase radar under ISRJ retransmission "
                 f"jamming (τ_j = {TAU_J_US:g} μs, J/S = {JSR_DB:g} dB, "
                 f"SNR = {SNR_DB:g} dB)",
                 color=TEXT_PRIMARY, fontsize=11)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    ax.grid(True, color="#e4e2dd", lw=0.5)
    ax.legend(frameon=False, loc="upper right")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    out = OUTDIR / ("scene02_jamming_range_profile.png" if JSR_DB == 10.0
                    else f"scene02_jamming_range_profile_jsr{JSR_DB:g}dB.png")
    fig.savefig(out, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"\n图已保存：{out}")


if __name__ == "__main__":
    main()
