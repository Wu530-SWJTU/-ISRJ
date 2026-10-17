# -*- coding: utf-8 -*-
"""对抗场景：CFAR 检测（真目标 + 假目标全部检出、无虚警）。

口径（2026-10-01 用户定）：
    场景    = run_scene_jamming.py 正式口径（J/S=15 dB、SNR=10 dB、匹配滤波）
    CFAR    = CA-CFAR（幅度域）：双侧参考单元 N_ref、保护单元 N_guard
              门限 T_i = mean(参考单元幅度) × α，
              α = sqrt(−4·ln(Pfa)/π)（Rayleigh 噪声，幅度域 Pfa 换算）
    参数    = Pfa = 1e-6、N_ref = 16、N_guard = 24（挡住真目标主瓣/旁瓣入参考窗）
    凝聚    = 检出连续段取峰值；距真目标/已知假目标 ≤ 0.2 km 的段归属该目标，
              其余记虚警（真目标距离旁瓣在 ±0.2 km 内，被凝聚吸收）

用法：
    .venv/Scripts/python.exe 参数搜索/scripts/run_scene_cfar.py
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

from run_scene_jamming import (  # noqa: E402
    R_T_KM, WIN_KM, TAU_T_US, KM_PER_US,
    scene_output, theoretical_false_targets, TEXT_PRIMARY, TEXT_SECONDARY,
    SURFACE, OUTDIR, NPZ,
)
import baseline  # noqa: E402

PFA = 1e-6                 # 虚警率（用户要求"无虚警" → 取 1e-6）
N_REF = 16                 # 单侧参考单元数
N_GUARD = 24               # 单侧保护单元数（真目标旁瓣/主瓣不进参考窗）
MERGE_KM = 0.2             # 峰值凝聚半径（真目标旁瓣在此半径内）

# 幅度域 CA-CFAR 门限因子：T = m̂·α，Rayleigh 噪声 Pfa = exp(−T²/(2σ²))，
# m̂ = σ√(π/2) ⇒ α = √(−4·ln(Pfa)/π)
ALPHA = np.sqrt(-4 * np.log(PFA) / np.pi)


def cfar_threshold(y_lin: np.ndarray, n_ref: int, n_guard: int) -> np.ndarray:
    """滑动 CA-CFAR：返回每个点的门限（幅度），边界外为 +inf。"""
    n = len(y_lin)
    t = np.full(n, np.inf)
    for i in range(n):
        lo = max(0, i - n_guard - n_ref)
        hi = min(n, i + n_guard + n_ref + 1)
        ref = np.concatenate([
            y_lin[lo:max(lo, i - n_guard)],
            y_lin[min(hi, i + n_guard + 1):hi]])
        if len(ref) >= n_ref:
            t[i] = ref.mean() * ALPHA
    return t


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    OUTDIR.mkdir(parents=True, exist_ok=True)

    x = baseline.lfm_waveform()      # 正常雷达 = LFM（2026-10-01 用户纠正）
    from isrj_joint_design.params import generate_isrj_sampling_function
    j = generate_isrj_sampling_function(baseline.params_for())
    tau_t_samples = int(round(TAU_T_US / baseline.SAMPLE_US))
    r_km, y_db = scene_output(x, j, tau_t_samples)

    m = (r_km >= R_T_KM + WIN_KM[0]) & (r_km <= R_T_KM + WIN_KM[1])
    rk, yk_lin = r_km[m], 10 ** (y_db[m] / 20)

    # ---- CFAR 门限与检出 ----
    thr_lin = cfar_threshold(yk_lin, N_REF, N_GUARD)
    thr_db = 20 * np.log10(thr_lin + 1e-30)
    det = yk_lin > thr_lin

    # ---- 连续段凝聚 → 归属 ----
    true_positions = [(R_T_KM, "true target")]
    for m_, lag_us, dkm, amp_db in theoretical_false_targets(x, j):
        true_positions.append((R_T_KM + dkm, f"m={m_:+d}"))

    segs, start, in_seg = [], None, False
    for i, dd in enumerate(det):
        if dd and start is None:
            start = i
        elif not dd and start is not None:
            segs.append((start, i))
            start = None
    if start is not None:
        segs.append((start, len(det)))

    print(f"CFAR 参数：Pfa={PFA:g}、N_ref={N_REF}、N_guard={N_GUARD}、"
          f"门限因子 α={ALPHA:.2f}（{20 * np.log10(ALPHA):.1f} dB）")
    print(f"（噪声底 −34 dB，幅度域均值 −32.1 dB ⇒ 门限 ≈ "
          f"{-32.1 + 20 * np.log10(ALPHA):.1f} dB）\n")

    detections, false_alarms = [], []
    print("=== 检测结果（连续段 → 峰值 → 归属）===")
    for a, b in segs:
        i_peak = a + int(np.argmax(yk_lin[a:b]))
        pos, lvl = rk[i_peak], 20 * np.log10(yk_lin[i_peak])
        owner = None
        for tp, name in true_positions:
            if abs(pos - tp) <= MERGE_KM:
                owner = name
                break
        if owner:
            if owner not in [x[0] for x in detections]:
                detections.append((owner, pos, lvl))
        else:
            false_alarms.append((pos, lvl))
        tag = f"→ {owner}" if owner else "→ ⚠ 虚警"
        print(f"   段 [{rk[a]:7.3f}, {rk[b - 1]:7.3f}] km  峰值 {pos:7.3f} km  "
              f"{lvl:+6.2f} dB   {tag}")

    print(f"\n检出目标 {len(detections)} 个：")
    for name, pos, lvl in sorted(detections, key=lambda v: v[1]):
        print(f"   {name:14s} @ {pos:7.3f} km   {lvl:+6.2f} dB")
    print(f"虚警 {len(false_alarms)} 个")

    # ---- 绘图 ----
    fig, ax = plt.subplots(figsize=(11, 5.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.plot(rk, y_db[m], color="#2a6f97", lw=0.9, zorder=2,
            label="matched filter output")
    ax.plot(rk, thr_db, color="#d1495b", lw=0.9, zorder=3,
            label=f"CFAR threshold (Pfa = {PFA:g})")
    ax.scatter([p for _, p, _ in detections], [l for _, _, l in detections],
               color="#2e7d32", s=55, zorder=5, edgecolor="white",
               linewidth=0.6, label=f"detections ({len(detections)})")
    if false_alarms:
        ax.scatter([p for p, _ in false_alarms], [l for _, l in false_alarms],
                   color="red", marker="x", s=70, zorder=6, label="false alarms")
    ax.axvline(R_T_KM, color="#e4572e", lw=0.9, ls="--", alpha=0.7)
    ax.set_xlim(R_T_KM + WIN_KM[0], R_T_KM + WIN_KM[1])
    ax.set_ylim(-60, 5)
    ax.set_xlabel("Range (km)", color=TEXT_SECONDARY)
    ax.set_ylabel("Normalized level (dB)", color=TEXT_SECONDARY)
    ax.set_title("CFAR detection — ISRJ jamming scene "
                 "(J/S = 15 dB, SNR = 10 dB; CA-CFAR, all targets detected, "
                 "no false alarm)", color=TEXT_PRIMARY, fontsize=11)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    ax.grid(True, color="#e4e2dd", lw=0.5)
    ax.legend(frameon=False, loc="upper right")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    out = OUTDIR / "scene03_cfar_detection.png"
    fig.savefig(out, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"\n图已保存：{out}")


if __name__ == "__main__":
    main()
