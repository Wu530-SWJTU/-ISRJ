# -*- coding: utf-8 -*-
"""对抗场景第 4 步（扩展）：三个干扰参数的误差扫描。

参数（沿用搜索线 v2 基线，采样点）：
    t_0 = 20（窗口起点 2 μs）、T_L = 15（截取长度 1.5 μs）、T_s = 64（周期 6.4 μs）
    K = 4 片，R = 3 次重复转发

误差档位：Δ ∈ {−5, −1, 0, +1, +5} 个采样点（1 点 = 0.1 μs）

口径：
    - 设计按**精确参数**做（`scene_joint_design`），x、h 固定不变
    - 评估时干扰机用**真实参数** θ* = θ̂ + Δ 生成：
        采样函数 j(θ*)     —— 窗口位置随 t_0 移、片长随 T_L 变、周期随 T_s 变
        转发时延 r·T_L*    —— 片长变了，转发时延跟着变 ⇒ **假目标位置也变**
    - 指标：**干扰单独**经 h 在真实假目标位置的电平（dB，相对主瓣）
      = 设计口径（不含 J/S；场景口径再加 J/S = 10 dB）
    - 占空比约束 (1+R)·T_L ≤ T_s：T_L 偏大时自动缩减 R（物理上转不下）

用法：
    .venv/Scripts/python.exe 参数搜索/scripts/run_scene_error_sweep_3param.py
"""

from __future__ import annotations

import dataclasses
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
from run_scene_jamming import (  # noqa: E402
    JSR_DB, KM_PER_US, R_T_KM, SEED, SNR_DB, TAU_T_US, TEXT_PRIMARY,
    TEXT_SECONDARY, SURFACE,
)
from scene_joint_design import joint_design_scene  # noqa: E402

OUTDIR = ROOT / "参数搜索" / "结果" / "对抗场景"
M_OUTER = 100
DELTAS = [-5, -1, 0, 1, 5]
# 场景口径的**唯一来源**是 `baseline.py`（与参数调制的项目参数同源），
# 这里只做"μs → 采样点"的换算，不再硬编码数值。
BASE = {"t_0": int(round(baseline.T0_US / baseline.SAMPLE_US)),
        "T_L": int(round(baseline.TAU_US / baseline.SAMPLE_US)),
        "T_s": int(round(baseline.TS_US / baseline.SAMPLE_US))}      # 采样点
N_SLICES = 4
R_MAX = BASE["T_s"] // BASE["T_L"] - 1        # 占空比约束 (1+R)·T_L ≤ T_s


def build_mask(n_off: int, n_tau: int, n_ts: int, N: int = 256) -> np.ndarray:
    """按 (窗口起点, 片长, 周期) 构造 ISRJ 采样函数（与项目函数逐位一致）。"""
    j = np.zeros(N)
    for s in range(N_SLICES):
        st = n_off + s * n_ts
        if st >= N:
            break
        j[st:min(st + n_tau, N)] = 1.0
    return j


def design_for(theta_hat: dict) -> tuple:
    """按雷达的**假设参数** θ̂ 做联合设计，返回 (x, h)。

    关键设定（2026-10-01 用户确认）：**雷达的参数估计有误差，干扰机始终是基线
    参数**。所以每档误差都要重新设计一次——雷达用错的 θ̂ 造掩膜、算转发时延，
    再把零陷钉在错的 lag 上。
    """
    j_hat = build_mask(theta_hat["t_0"], theta_hat["T_L"], theta_hat["T_s"])
    R_hat = max(min(R_MAX, theta_hat["T_s"] // theta_hat["T_L"] - 1), 1)
    offs_hat = [r * theta_hat["T_L"] for r in range(1, R_hat + 1)]
    res = joint_design_scene(dataclasses.replace(baseline.params_for(), M=M_OUTER),
                             j_hat, offs_hat, [0] + offs_hat, verbose=False)
    return res["x"], res["h"]


def scene_profile(x: np.ndarray, h: np.ndarray, j_actual: np.ndarray,
                  offsets: list, tau_t_samples: int) -> tuple:
    """雷达实际看到的距离像（真目标 + 干扰含 J/S + 噪声），返回 (距离 km, dB)。"""
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
    return r_km, 20 * np.log10(np.abs(y) / ref + 1e-30)


def jamming_profile_scene(x: np.ndarray, h: np.ndarray, j_actual: np.ndarray,
                          offsets: list, tau_t_samples: int) -> tuple:
    """干扰单独经 h（含 J/S，无真目标无噪声），同一距离轴与归一化。"""
    n = len(x)
    a_j = 10 ** (JSR_DB / 20)
    r = np.zeros(tau_t_samples + max(offsets) + n, dtype=complex)
    for d in offsets:
        s = tau_t_samples + d
        r[s:s + n] += a_j * (x * j_actual)

    y = np.convolve(r, np.conj(h[::-1]), mode="full")
    peak_idx = tau_t_samples + n - 1
    ref = abs(np.vdot(h, x))
    r_km = R_T_KM + (np.arange(len(y)) - peak_idx) * baseline.SAMPLE_US * KM_PER_US
    return r_km, 20 * np.log10(np.abs(y) / ref + 1e-30)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    OUTDIR.mkdir(parents=True, exist_ok=True)

    j_true = build_mask(BASE["t_0"], BASE["T_L"], BASE["T_s"])
    assert np.abs(j_true - generate_isrj_sampling_function(
        baseline.params_for())).max() == 0, "掩膜构造与项目函数不一致"
    tau_t = int(round(TAU_T_US / baseline.SAMPLE_US))
    TRUE_OFFSETS = [r * BASE["T_L"] for r in range(1, R_MAX + 1)]

    print("设定：**雷达的参数估计有误差，干扰机始终是基线参数**")
    print(f"  真实干扰机（固定不变）：t_0={BASE['t_0']}、T_L={BASE['T_L']}、"
          f"T_s={BASE['T_s']} 点，R={R_MAX}")
    print(f"  假目标固定在 +1.5/+3.0/+4.5 μs（= +225/+450/+675 m）")
    print(f"  每档误差都**重新设计一次**（雷达用错的 θ̂ = θ* + Δ 造掩膜、钉零陷）\n")

    # ---- 三参数扫描（每档重新设计，对固定的真实干扰机评估）----
    x_ref, h_ref = design_for(dict(BASE))        # Δ=0 的设计（三行共用）
    table = {}
    for param in BASE:
        table[param] = {}
        for delta in DELTAS:
            if delta == 0:
                x, h = x_ref, h_ref
            else:
                theta_hat = dict(BASE)
                theta_hat[param] = BASE[param] + delta
                x, h = design_for(theta_hat)
            r_km, y_jam = jamming_profile_scene(x, h, j_true, TRUE_OFFSETS, tau_t)
            table[param][delta] = [
                float(y_jam[int(np.argmin(np.abs(
                    r_km - (R_T_KM + o * baseline.SAMPLE_US * KM_PER_US))))])
                for o in TRUE_OFFSETS]
        print(f"  {param} 扫描完成")

    # ---- 打印表 ----
    print(f"\n{'参数':<6}{'Δ(点)':>7}   三个假目标电平 (dB)              最差")
    for param in BASE:
        for delta in DELTAS:
            lv = table[param][delta]
            print(f"{param:<6}{delta:>7d}   {'  '.join('%7.2f' % v for v in lv)}"
                  f"   {max(lv):7.2f}")
    print()

    # ---- 绘图：分组柱状图 ----
    labels = [f"Δ={d:+d}" for d in DELTAS]
    xpos = np.arange(len(DELTAS))
    width = 0.26
    fig, ax = plt.subplots(figsize=(10, 5.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    colors = {"t_0": "#1f4e79", "T_L": "#2e7d32", "T_s": "#c0392b"}
    names = {"t_0": "$t_0$ (window start)", "T_L": "$T_L$ (slice length)",
             "T_s": "$T_s$ (period)"}
    for i, param in enumerate(BASE):
        vals = [max(table[param][d]) for d in DELTAS]   # 三个假目标的最差电平
        bars = ax.bar(xpos + (i - 1) * width, vals, width, color=colors[param],
                      label=names[param], zorder=3)
        for b, v in zip(bars, vals):
            ax.annotate(f"{v:.1f}", (b.get_x() + b.get_width() / 2, v),
                        ha="center", va="bottom", fontsize=7.5,
                        color=TEXT_SECONDARY)
    ax.axhline(-70, color="#7f8c8d", lw=1.1, ls="-.", zorder=2,
               label="design level (exact): −70 dB")
    ax.axhline(-2.6, color="#c9a227", lw=1.1, ls="--", zorder=2,
               label="no design: −2.6 dB")
    ax.set_xticks(xpos)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Parameter estimation error (sampling points; 1 point = 0.1 μs)",
                  color=TEXT_SECONDARY)
    ax.set_ylabel("Worst false target level (dB, rel. mainlobe)", color=TEXT_SECONDARY)
    ax.set_title("Suppression vs parameter error — $t_0$ / $T_L$ / $T_s$ "
                 "(design caliber, J/S excluded)", color=TEXT_PRIMARY, fontsize=11)
    ax.set_ylim(-85, 5)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    ax.grid(True, axis="y", color="#e4e2dd", lw=0.5)
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # ---- 幅度-距离图：3 参数 × 3 档 ----
    tau_t = int(round(TAU_T_US / baseline.SAMPLE_US))
    sigma2 = 10 ** (-SNR_DB / 10)
    noise_floor_db = float(20 * np.log10(
        np.sqrt(sigma2 * np.sum(np.abs(h_ref) ** 2)) / abs(np.vdot(h_ref, x_ref))))
    panels = [("t_0", 0), ("t_0", 1), ("t_0", 5),
              ("T_L", 0), ("T_L", 1), ("T_L", 5),
              ("T_s", 0), ("T_s", 1), ("T_s", 5)]
    pnames = {"t_0": "$t_0$", "T_L": "$T_L$", "T_s": "$T_s$"}
    fig2, axes = plt.subplots(3, 3, figsize=(14.5, 10.5), facecolor=SURFACE)
    win_km = (R_T_KM - 0.35, R_T_KM + 0.95)
    for ax, (param, delta) in zip(axes.ravel(), panels):
        ax.set_facecolor(SURFACE)
        theta_hat = dict(BASE)
        theta_hat[param] = BASE[param] + delta
        x, h = design_for(theta_hat)          # 雷达用错的 θ̂ 设计
        offs = TRUE_OFFSETS                   # 干扰机不变 → 假目标位置固定

        r_km, y_tot = scene_profile(x, h, j_true, offs, tau_t)
        m = (r_km >= win_km[0]) & (r_km <= win_km[1])
        ax.plot(r_km[m], y_tot[m], color="#2a6f97", lw=0.9, zorder=2,
                label="range profile (target + jamming + noise)")
        ax.axhline(noise_floor_db, color="#7f8c8d", lw=1.0, ls="-.", zorder=2,
                   label=f"output noise floor {noise_floor_db:.1f} dB")
        ax.axvline(R_T_KM, color="#e4572e", lw=0.9, ls="--", zorder=3)
        for o in offs:
            p = R_T_KM + o * baseline.SAMPLE_US * KM_PER_US
            lvl = float(y_tot[int(np.argmin(np.abs(r_km - p)))])
            ax.axvline(p, color="#c9a227", lw=0.7, ls=":", zorder=3)
            ax.scatter([p], [lvl], color="#c9a227", s=20, zorder=5,
                       edgecolor="white", linewidth=0.5)
            ax.annotate(f"{lvl:+.1f} dB", (p, lvl), xytext=(0, 5),
                        textcoords="offset points", ha="center", va="bottom",
                        fontsize=7.5, color="#8a6d1a", zorder=6)
        ax.set_ylim(-60, 6)
        ax.set_title(f"{pnames[param]}  Δ = {delta:+d} sample  (radar's "
                     f"estimate wrong; jammer unchanged)",
                     color=TEXT_PRIMARY, fontsize=9)
        ax.tick_params(colors=TEXT_SECONDARY, labelsize=8)
        ax.grid(True, color="#e4e2dd", lw=0.4)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        if delta == 0:
            ax.legend(frameon=False, loc="upper left", fontsize=7.5)
    for ax in axes[-1]:
        ax.set_xlabel("Range (km)", color=TEXT_SECONDARY, fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel("Level (dB)", color=TEXT_SECONDARY, fontsize=9)
    fig2.suptitle("Range profiles under parameter mismatch — "
                  "$t_0$ / $T_L$ / $T_s$  (self-screening ISRJ, J/S = 10 dB, "
                  "SNR = 10 dB)", color=TEXT_PRIMARY, fontsize=12)
    out_prof = OUTDIR / "scene05_error_3param_profiles.png"
    fig2.savefig(out_prof, dpi=140, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig2)
    print(f"三参数距离像已保存：{out_prof}")

    out = OUTDIR / "scene05_error_3param.png"
    fig.savefig(out, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)

    (OUTDIR / "scene05_error_3param.json").write_text(json.dumps({
        "base_samples": BASE, "deltas": DELTAS,
        "levels_db_scene_caliber": {p: {str(d): v for d, v in table[p].items()}
                                    for p in BASE},
        "true_jammer": dict(BASE), "true_offsets_samples": TRUE_OFFSETS,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"图已保存：{out}")
    print(f"数据已保存：{OUTDIR / 'scene05_error_3param.json'}")


if __name__ == "__main__":
    main()
