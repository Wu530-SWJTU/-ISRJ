# -*- coding: utf-8 -*-
"""对抗场景第 4 步（联合）：三个参数同时带误差。

误差向量 Δ = (Δt_0, ΔT_L, ΔT_s)，单位采样点（1 点 = 0.1 μs）。
设计的 (x, h) 固定不变（按精确参数设计）；评估时干扰机用真实参数 θ̂+Δ 生成。

口径同 `docs/对抗场景口径.md`：随机相位波形、自卫干扰、R ≤ 3、J/S=10 dB、SNR=10 dB。
指标 = 干扰单独经 h 在**真实**假目标位置的电平（设计口径，不含 J/S）。

用法：
    .venv/Scripts/python.exe 参数搜索/scripts/run_scene_error_joint.py
"""

from __future__ import annotations

import json
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
from run_scene_error_sweep_3param import (  # noqa: E402
    BASE, R_MAX, build_mask, design_for, jamming_profile_scene, scene_profile,
)
from run_scene_jamming import (  # noqa: E402
    JSR_DB, KM_PER_US, R_T_KM, SEED, SNR_DB, TAU_T_US, TEXT_PRIMARY,
    TEXT_SECONDARY, SURFACE,
)

OUTDIR = ROOT / "参数搜索" / "结果" / "对抗场景"
M_OUTER = 100

# (标签, Δt_0, ΔT_L, ΔT_s)  —— 单位：采样点
CASES = [
    ("exact", 0, 0, 0),
    ("all +1", 1, 1, 1),
    ("all −1", -1, -1, -1),
    ("all +5", 5, 5, 5),
    ("all −5", -5, -5, -5),
    ("(+1,−1,+1)", 1, -1, 1),      # 检验正负误差是否互相抵消
    ("(−1,+1,−1)", -1, 1, -1),
    ("(+1,+1,−1)", 1, 1, -1),
]


def design_case(dt0: int, dtl: int, dts: int) -> tuple:
    """雷达按**错误的估计** θ̂ = θ* + Δ 设计，返回 (x, h)。

    干扰机始终是基线参数（用户 2026-10-01 确认的设定）。
    """
    theta_hat = {"t_0": BASE["t_0"] + dt0, "T_L": BASE["T_L"] + dtl,
                 "T_s": BASE["T_s"] + dts}
    return design_for(theta_hat)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    OUTDIR.mkdir(parents=True, exist_ok=True)

    j_true = build_mask(BASE["t_0"], BASE["T_L"], BASE["T_s"])
    assert np.abs(j_true - generate_isrj_sampling_function(
        baseline.params_for())).max() == 0, "掩膜构造与项目函数不一致"
    offs_true = [r * BASE["T_L"] for r in range(1, R_MAX + 1)]
    tau_t = int(round(TAU_T_US / baseline.SAMPLE_US))
    x_ref, h_ref = design_case(0, 0, 0)
    noise_floor_db = float(20 * np.log10(
        np.sqrt(10 ** (-SNR_DB / 10) * np.sum(np.abs(h_ref) ** 2))
        / abs(np.vdot(h_ref, x_ref))))

    print("设定：**雷达估计有误差，干扰机始终是基线参数**")
    print(f"  真实干扰机（固定）：t_0={BASE['t_0']}、T_L={BASE['T_L']}、"
          f"T_s={BASE['T_s']} 点，R={R_MAX}，假目标固定 +1.5/+3.0/+4.5 μs")
    print(f"  每档误差重新设计一次；噪声底 {noise_floor_db:.2f} dB\n")

    print(f"{'误差向量 (Δt0,ΔTL,ΔTs)':<24}  三个假目标电平 (dB)            最差")
    table = {}
    for label, dt0, dtl, dts in CASES:
        x, h = design_case(dt0, dtl, dts)
        r_km, y_jam = jamming_profile_scene(x, h, j_true, offs_true, tau_t)
        lv = [float(y_jam[int(np.argmin(np.abs(
            r_km - (R_T_KM + o * baseline.SAMPLE_US * KM_PER_US))))])
            for o in offs_true]
        table[label] = {"delta": [dt0, dtl, dts], "levels_db": lv}
        print(f"{label:<24} {'  '.join('%7.2f' % v for v in lv):<28} {max(lv):7.2f}")

    # ---- 图：精确 / 全 +1 / 全 +5 ----
    panels = [("exact parameters", (0, 0, 0)),
              ("all three +1 sample (0.1 μs)", (1, 1, 1)),
              ("all three +5 samples (0.5 μs)", (5, 5, 5))]
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.4), facecolor=SURFACE)
    win_km = (R_T_KM - 0.35, R_T_KM + 0.95)
    for ax, (title, (dt0, dtl, dts)) in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        x, h = design_case(dt0, dtl, dts)
        offs = offs_true
        r_km, y_tot = scene_profile(x, h, j_true, offs, tau_t)
        m = (r_km >= win_km[0]) & (r_km <= win_km[1])
        ax.plot(r_km[m], y_tot[m], color="#2a6f97", lw=1.0, zorder=2,
                label="range profile (target + jamming + noise)")
        ax.axhline(noise_floor_db, color="#7f8c8d", lw=1.0, ls="-.", zorder=2,
                   label=f"noise floor {noise_floor_db:.1f} dB")
        ax.axvline(R_T_KM, color="#e4572e", lw=1.0, ls="--", zorder=3)
        for o in offs:
            p = R_T_KM + o * baseline.SAMPLE_US * KM_PER_US
            lvl = float(y_tot[int(np.argmin(np.abs(r_km - p)))])
            ax.axvline(p, color="#c9a227", lw=0.8, ls=":", zorder=3)
            ax.scatter([p], [lvl], color="#c9a227", s=26, zorder=5,
                       edgecolor="white", linewidth=0.5)
            ax.annotate(f"{lvl:+.1f} dB", (p, lvl), xytext=(0, 5),
                        textcoords="offset points", ha="center", va="bottom",
                        fontsize=7.5, color="#8a6d1a", zorder=6)
        ax.set_ylim(-60, 6)
        ax.set_xlim(*win_km)
        ax.set_title(title, color=TEXT_PRIMARY, fontsize=10.5)
        ax.set_xlabel("Range (km)", color=TEXT_SECONDARY, fontsize=9)
        ax.tick_params(colors=TEXT_SECONDARY, labelsize=8.5)
        ax.grid(True, color="#e4e2dd", lw=0.4)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("Level (dB)", color=TEXT_SECONDARY, fontsize=9)
    axes[0].legend(frameon=False, loc="upper left", fontsize=8)
    fig.suptitle("Range profiles under SIMULTANEOUS $t_0$/$T_L$/$T_s$ mismatch "
                 "(self-screening ISRJ, J/S = 10 dB, SNR = 10 dB)",
                 color=TEXT_PRIMARY, fontsize=12)
    out = OUTDIR / "scene06_error_joint_profiles.png"
    fig.savefig(out, dpi=140, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)

    (OUTDIR / "scene06_error_joint.json").write_text(
        json.dumps(table, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n图已保存：{out}")
    print(f"数据已保存：{OUTDIR / 'scene06_error_joint.json'}")


if __name__ == "__main__":
    main()
