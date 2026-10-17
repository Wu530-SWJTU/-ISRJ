# -*- coding: utf-8 -*-
"""对抗场景第 1 步（2026-10-01 重做）：未优化随机相位波形 + 匹配滤波正常探测，
加噪声，画幅度-距离图。

口径（用户 2026-10-01 定）：
    波形   = 随机相位波形 exp(jφ_n)，未优化，匹配滤波（fs = B，N = 256）
    真目标 = 30 km（时延 30 μs），幅度 1（0 dB 参考 = 波形能量 Σ|x|² = 256）
    噪声   = 复高斯白噪声，匹配滤波前 SNR = 10 dB（σ² = 0.1/采样点），seed = 1
    纵轴   = 归一化电平 dB；横轴 = 距离 km（1 μs = 150 m）
    本步无干扰。不做 CFAR 检测（2026-10-01 用户定）。

用法：
    .venv/Scripts/python.exe 参数搜索/scripts/run_scene_normal.py
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

OUTDIR = ROOT / "参数搜索" / "结果" / "对抗场景"
TAU_T_US = 30.0            # 真目标时延（μs）
R_T_KM = 30.0              # 真目标距离（km），c·τ/2 = 150 m/μs
KM_PER_US = 0.15
WIN_KM = (-4.0, 4.0)       # 图窗：真目标 ±4 km
SNR_DB = 10.0              # 匹配滤波前 SNR（2026-10-01 定）
SEED = 1                   # 噪声种子（可复现）
TEXT_PRIMARY, TEXT_SECONDARY = "#0b0b0b", "#52514e"
SURFACE = "#fcfcfb"


def matched_filter_output(x: np.ndarray, tau_t_samples: int) -> tuple:
    """接收 = 延迟回波 + 白噪声；匹配滤波输出与距离轴（km）。

    真目标峰位置由场景几何解析给出（不查 argmax）；归一化参考 = Σ|x|²。
    """
    n = len(x)
    r = np.zeros(tau_t_samples + n, dtype=complex)
    r[tau_t_samples:tau_t_samples + n] += x
    sigma = 10 ** (-SNR_DB / 20)
    rng = np.random.default_rng(SEED)
    r += (rng.standard_normal(len(r)) + 1j * rng.standard_normal(len(r))) \
        * (sigma / np.sqrt(2))

    y = np.convolve(r, np.conj(x[::-1]), mode="full")
    peak_idx = tau_t_samples + n - 1
    ref = (np.abs(x) ** 2).sum()
    r_km = R_T_KM + (np.arange(len(y)) - peak_idx) * baseline.SAMPLE_US * KM_PER_US
    y_db = 20 * np.log10(np.abs(y) / ref + 1e-30)
    return r_km, y_db


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    OUTDIR.mkdir(parents=True, exist_ok=True)

    x = baseline.random_phase_waveform()   # 未优化随机相位波形（2026-10-01 用户改）
    tau_t_samples = int(round(TAU_T_US / baseline.SAMPLE_US))
    r_km, y_db = matched_filter_output(x, tau_t_samples)

    m = (r_km >= R_T_KM + WIN_KM[0]) & (r_km <= R_T_KM + WIN_KM[1])
    rk = r_km[m]
    yk_db = y_db[m]

    print(f"场景：随机相位波形（未优化）+ 匹配滤波，SNR = {SNR_DB:g} dB（σ² = "
          f"{10 ** (-SNR_DB / 10):.2g}/采样点），seed = {SEED}；本步无干扰")
    print(f"真目标峰：{yk_db.max():+.2f} dB @ "
          f"{rk[int(np.argmax(yk_db))]:.3f} km")

    # ---- 绘图 ----
    fig, ax = plt.subplots(figsize=(11, 5.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.plot(rk, yk_db, color="#2a6f97", lw=0.9, zorder=2,
            label="matched filter output")
    ax.axvline(R_T_KM, color="#e4572e", lw=0.9, ls="--", zorder=3,
               label="true target (30 km)")
    ax.axhline(0.0, color=TEXT_SECONDARY, lw=0.5, ls=":")
    ax.set_xlim(R_T_KM + WIN_KM[0], R_T_KM + WIN_KM[1])
    ax.set_ylim(-60, 5)
    ax.set_xlabel("Range (km)", color=TEXT_SECONDARY)
    ax.set_ylabel("Normalized level (dB)", color=TEXT_SECONDARY)
    ax.set_title("Step 1 — unoptimized random-phase radar, matched filter, "
                 f"target echo only (SNR = {SNR_DB:g} dB)",
                 color=TEXT_PRIMARY, fontsize=11)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    ax.grid(True, color="#e4e2dd", lw=0.5)
    ax.legend(frameon=False, loc="upper right")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    out = OUTDIR / "scene01_normal_range_profile.png"
    fig.savefig(out, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"\n图已保存：{out}")


if __name__ == "__main__":
    main()
