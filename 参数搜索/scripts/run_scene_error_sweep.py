# -*- coding: utf-8 -*-
"""对抗场景第 4 步：参数误差下的压制效果（误差扫描曲线）。

口径（2026-10-01 用户定）：
    - 设计：`scene_joint_design`，按**精确已知**的干扰参数 θ̂ = θ* 设计 (x, h)
    - 误差：只扰 **t_0（窗口起点）**，Δ ∈ {0, ±1, ..., ±8} 个采样点
      （1 采样点 = 0.1 μs；**小于 0.5 点的误差不改变采样函数 j，图不变**）
    - 假目标位置由转发时延决定（r·T_L），与 t_0 误差无关 ⇒ 仍读同样的三个位置
    - 参考线：**不做设计**（随机相位 + 匹配滤波）时的假目标电平

干扰模型（经典 ISRJ 重复转发）：
    y_jam = Σ_{r=1}^{R} shift_{r·T_L}( x ⊙ j_actual )，j_actual = shift_{Δ}(j)

用法：
    .venv/Scripts/python.exe 参数搜索/scripts/run_scene_error_sweep.py
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
from run_scene_error_sweep_3param import design_for  # noqa: E402
from run_scene_jamming import (  # noqa: E402
    JSR_DB, KM_PER_US, R_T_KM, SEED, SNR_DB, TAU_T_US, TEXT_PRIMARY,
    TEXT_SECONDARY, SURFACE,
)
from scene_joint_design import R_RETX, T_L_US  # noqa: E402

OUTDIR = ROOT / "参数搜索" / "结果" / "对抗场景"
M_OUTER = 100
DELTAS = list(range(-8, 9))       # 误差档位（采样点）


def range_profile(x: np.ndarray, h: np.ndarray, j_actual: np.ndarray,
                  offsets: list, tau_t_samples: int) -> tuple:
    """给定 (x,h) 与实际采样函数，返回距离像 (距离 km, 电平 dB)。"""
    n = len(x)
    a_j = 10 ** (JSR_DB / 20)
    r = np.zeros(tau_t_samples + max(offsets) + n, dtype=complex)
    r[tau_t_samples:tau_t_samples + n] += x
    for d in offsets:
        s = tau_t_samples + d
        r[s:s + n] += a_j * (x * j_actual)

    sigma = 10 ** (-SNR_DB / 20)
    rng = np.random.default_rng(SEED)
    r += (rng.standard_normal(len(r)) + 1j * rng.standard_normal(len(r))) \
        * (sigma / np.sqrt(2))

    y = np.convolve(r, np.conj(h[::-1]), mode="full")
    peak_idx = tau_t_samples + n - 1
    ref = abs(np.vdot(h, x))
    r_km = R_T_KM + (np.arange(len(y)) - peak_idx) * baseline.SAMPLE_US * KM_PER_US
    y_db = 20 * np.log10(np.abs(y) / ref + 1e-30)
    return r_km, y_db


def measure(x: np.ndarray, h: np.ndarray, j_actual: np.ndarray,
            offsets: list, tau_t_samples: int) -> list:
    """三个假目标处的电平（dB，相对主峰）。"""
    r_km, y_db = range_profile(x, h, j_actual, offsets, tau_t_samples)
    peak_idx = tau_t_samples + len(x) - 1
    return [float(y_db[peak_idx + d]) for d in offsets]


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    OUTDIR.mkdir(parents=True, exist_ok=True)

    j = generate_isrj_sampling_function(baseline.params_for())
    d_samples = int(round(T_L_US / baseline.SAMPLE_US))
    offsets = [r * d_samples for r in range(1, R_RETX + 1)]
    tau_t_samples = int(round(TAU_T_US / baseline.SAMPLE_US))
    base_theta = {"t_0": int(round(baseline.T0_US / baseline.SAMPLE_US)),
                  "T_L": int(round(baseline.TAU_US / baseline.SAMPLE_US)),
                  "T_s": int(round(baseline.TS_US / baseline.SAMPLE_US))}

    print("设定：**雷达对 t_0 的估计有误差，干扰机始终是基线参数**")
    print(f"  真实干扰机（固定）：窗口起点 {base_theta['t_0']} 点、T_L="
          f"{base_theta['T_L']}、T_s={base_theta['T_s']}；假目标固定 +1.5/+3.0/+4.5 μs")
    print("  每档误差重新设计一次\n")

    # 参考：不做设计（随机相位 + 匹配滤波），干扰机仍为基线
    x0 = baseline.random_phase_waveform()
    before = measure(x0, x0, j, offsets, tau_t_samples)
    before_worst = max(before)
    print(f"无设计参考（匹配滤波）：假目标 {['%.2f' % v for v in before]} dB\n")

    # 扫描 t_0 估计误差（每档用错的 θ̂ 重新设计，对固定干扰机评估）
    print(f"{'Δ(采样点)':>10} {'Δ(μs)':>8}   假目标电平 (dB)")
    sweep, ref_pair = {}, None
    for delta in DELTAS:
        theta_hat = dict(base_theta)
        theta_hat["t_0"] = base_theta["t_0"] + delta
        x_d, h_d = design_for(theta_hat)
        if delta == 0:
            ref_pair = (x_d, h_d)
        lv = measure(x_d, h_d, j, offsets, tau_t_samples)
        sweep[delta] = lv
        print(f"{delta:>10d} {delta * 0.1:>8.1f}   "
              f"{'  '.join('%7.2f' % v for v in lv)}   (最差 {max(lv):6.2f})")

    # ---- 绘图 ----
    xs = np.array(DELTAS, dtype=float)
    arr = np.array([sweep[k] for k in DELTAS])       # (nΔ, R)
    worst = arr.max(axis=1)

    # 输出噪声底（失配滤波 h 放大噪声：σ²||h||²/(h^H x)²）
    sigma2 = 10 ** (-SNR_DB / 10)
    x_ref, h_ref = ref_pair
    noise_floor_db = float(20 * np.log10(
        np.sqrt(sigma2 * np.sum(np.abs(h_ref) ** 2)) / abs(np.vdot(h_ref, x_ref))))
    pg_loss_db = float(-20 * np.log10(
        abs(np.vdot(h_ref, x_ref))
        / np.sqrt(len(x_ref) * np.sum(np.abs(h_ref) ** 2))))

    fig, ax = plt.subplots(figsize=(10.5, 5.8), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for r in range(len(offsets)):
        ax.plot(xs, arr[:, r], color="#9ec5d8", lw=1.0, zorder=2,
                label="false targets #1/#2/#3 (individual)" if r == 0 else None)
    ax.plot(xs, worst, color="#1f4e79", lw=2.2, marker="o", ms=4.2,
            zorder=3, label="worst false target (designed pair)")
    ax.axhline(before_worst, color="#c0392b", lw=1.3, ls="--", zorder=2,
               label=f"no design, matched filter: {before_worst:.1f} dB")
    ax.axhline(noise_floor_db, color="#7f8c8d", lw=1.1, ls="-.", zorder=2,
               label=f"output noise floor: {noise_floor_db:.1f} dB")
    ax.axvline(0, color="#b8b6b0", lw=0.8, ls=":", zorder=1)

    txt = (f"A  exact (Δ=0):      {worst[len(DELTAS) // 2]:+.1f} dB "
           f"— buried in noise floor\n"
           f"B  Δ=±1 sample:     {max(worst[len(DELTAS) // 2 - 1], worst[len(DELTAS) // 2 + 1]):+.1f} dB "
           f"— false targets back\n"
           f"C  Δ=±5 samples:   {max(worst[len(DELTAS) // 2 - 5], worst[len(DELTAS) // 2 + 5]):+.1f} dB "
           f"— suppression lost")
    ax.text(0.025, 0.035, txt, transform=ax.transAxes, fontsize=8.8,
            color="#1f4e79", va="bottom", ha="left",
            bbox=dict(boxstyle="round,pad=0.4", fc="#eaf2f8", ec="#1f4e79",
                      lw=0.7))

    ax.set_xlabel("$t_0$ estimation error (sampling points; 1 point = 0.1 μs)",
                  color=TEXT_SECONDARY)
    ax.set_ylabel("False target level (dB, rel. mainlobe)", color=TEXT_SECONDARY)
    ax.set_title("Suppression vs $t_0$ estimation error — ISRJ repeated "
                 f"retransmission (R={R_RETX}, J/S={JSR_DB:g} dB, "
                 f"SNR={SNR_DB:g} dB; mismatch filter costs {pg_loss_db:.2f} dB "
                 f"processing gain)", color=TEXT_PRIMARY, fontsize=10.5)
    ax.set_xlim(xs.min() - 0.5, xs.max() + 0.5)
    ax.set_ylim(-80, 2)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    ax.grid(True, color="#e4e2dd", lw=0.5)
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    out = OUTDIR / "scene04_error_sweep.png"
    fig.savefig(out, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)

    # ---- 幅度-距离图：Δ = 0 / ±1 / ±5 三档对比 ----
    panels = [(0, "A — radar's estimate exact (Δ = 0)"),
              (1, "B — $t_0$ estimate off by +1 sample (0.1 μs)"),
              (5, "C — $t_0$ estimate off by +5 samples (0.5 μs)")]
    fig2, axes = plt.subplots(len(panels), 1, figsize=(11, 9.5), sharex=True,
                              facecolor=SURFACE)
    win_km = (R_T_KM - 0.25, R_T_KM + 0.95)
    for ax, (delta, title) in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        theta_hat = dict(base_theta)
        theta_hat["t_0"] = base_theta["t_0"] + delta
        x_p, h_p = design_for(theta_hat)
        r_km, y_db = range_profile(x_p, h_p, j, offsets, tau_t_samples)
        m = (r_km >= win_km[0]) & (r_km <= win_km[1])
        ax.plot(r_km[m], y_db[m], color="#1f4e79", lw=1.0, zorder=2)
        ax.axhline(noise_floor_db, color="#7f8c8d", lw=1.0, ls="-.", zorder=2)
        ax.axvline(R_T_KM, color="#e4572e", lw=1.0, ls="--", zorder=3)
        ax.annotate("true\ntarget", (R_T_KM, 3), ha="center", va="top",
                    fontsize=8.5, color="#c0392b")
        for o in offsets:
            p = R_T_KM + o * baseline.SAMPLE_US * KM_PER_US
            lvl = float(y_db[int(np.argmin(np.abs(r_km - p)))])
            ax.axvline(p, color="#c9a227", lw=0.7, ls=":", zorder=3)
            ax.scatter([p], [lvl], color="#c9a227", s=26, zorder=5,
                       edgecolor="white", linewidth=0.5)
            ax.annotate(f"{lvl:+.1f} dB", (p, lvl), xytext=(0, 5),
                        textcoords="offset points", ha="center", va="bottom",
                        fontsize=7.5, color="#8a6d1a", zorder=6)
        ax.set_ylim(-60, 6)
        ax.set_ylabel("Level (dB)", color=TEXT_SECONDARY)
        ax.set_title(title, color=TEXT_PRIMARY, fontsize=10.5)
        ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
        ax.grid(True, color="#e4e2dd", lw=0.5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[-1].set_xlim(*win_km)
    axes[-1].set_xlabel("Range (km)", color=TEXT_SECONDARY)
    axes[0].plot([], [], color="#7f8c8d", ls="-.",
                 label=f"output noise floor {noise_floor_db:.1f} dB")
    axes[0].legend(frameon=False, loc="upper left", fontsize=9)
    fig2.suptitle("Range profiles under $t_0$ mismatch — false targets "
                  "reappear once parameters are mismatched",
                  color=TEXT_PRIMARY, fontsize=11.5)
    out2 = OUTDIR / "scene04_error_range_profiles.png"
    fig2.savefig(out2, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig2)
    print(f"三档距离像已保存：{out2}")

    (OUTDIR / "scene04_error_sweep.json").write_text(json.dumps({
        "offsets_samples": offsets,
        "deltas_points": DELTAS,
        "levels_db": {str(k): v for k, v in sweep.items()},
        "no_design_db": before,
        "jsr_db": JSR_DB, "snr_db": SNR_DB,
        "noise_floor_db": noise_floor_db,
        "processing_gain_loss_db": pg_loss_db,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n图已保存：{out}")
    print(f"数据已保存：{OUTDIR / 'scene04_error_sweep.json'}")


if __name__ == "__main__":
    main()
