# -*- coding: utf-8 -*-
"""对抗场景第 3 步：已知精确干扰下联合设计波形+失配滤波器，并做前后对比。

流程：
    ① 设计：`scene_joint_design.joint_design_scene`（R+1 条约束的交替优化）
    ② 前：随机相位波形 + 匹配滤波（当前场景口径）
    ③ 后：联合设计波形 x + 失配滤波 h（**闭环**：干扰机采样的是新波形 x）
    ④ 画对比图（上下两栏）

干扰模型（经典 ISRJ 重复转发，与 scene02 一致）：
    信号级  y_jam = Σ_{r=1}^{R} shift_{r·T_L}( x ⊙ j )
    矩阵链  Y_jam = ( Σ_r S_{r·T_L} ) · diag(j) · x
            脉压输出端 H_j = H · ( Σ_r S_{r·T_L} ) · diag(j)

用法：
    .venv/Scripts/python.exe 参数搜索/scripts/run_scene_design_eval.py
"""

from __future__ import annotations

import dataclasses
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
from scene_joint_design import (  # noqa: E402
    R_RETX, T_L_US, joint_design_scene,
)
from run_scene_jamming import (  # noqa: E402
    JSR_DB, KM_PER_US, R_T_KM, SEED, SNR_DB, TAU_T_US, WIN_KM,
    TEXT_PRIMARY, TEXT_SECONDARY, SURFACE,
)

OUTDIR = ROOT / "参数搜索" / "结果" / "对抗场景"
M_OUTER = 100               # 外层交替优化轮数（PRD 默认）


def range_profile(x: np.ndarray, h: np.ndarray, j: np.ndarray,
                  offsets: list, tau_t_samples: int) -> tuple:
    """接收（真目标 + R 份转发干扰 + 噪声）经滤波器 h 的距离像。"""
    n = len(x)
    a_j = 10 ** (JSR_DB / 20)
    hi = max(offsets)
    r = np.zeros(tau_t_samples + hi + n, dtype=complex)
    r[tau_t_samples:tau_t_samples + n] += x                       # 真目标回波
    for d in offsets:
        # 第 r 次转发：采样切片列车整体延迟 d（等价于 x⊙j 放在 tau_t+d）
        s = tau_t_samples + d
        r[s:s + n] += a_j * (x * j)

    sigma = 10 ** (-SNR_DB / 20)
    rng = np.random.default_rng(SEED)
    r += (rng.standard_normal(len(r)) + 1j * rng.standard_normal(len(r))) \
        * (sigma / np.sqrt(2))

    y = np.convolve(r, np.conj(h[::-1]), mode="full")
    peak_idx = tau_t_samples + n - 1
    ref = abs(np.vdot(h, x))                      # 主瓣 h^H x
    r_km = R_T_KM + (np.arange(len(y)) - peak_idx) * baseline.SAMPLE_US * KM_PER_US
    y_db = 20 * np.log10(np.abs(y) / ref + 1e-30)
    return r_km, y_db


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    OUTDIR.mkdir(parents=True, exist_ok=True)

    j = generate_isrj_sampling_function(baseline.params_for())
    d_samples = int(round(T_L_US / baseline.SAMPLE_US))
    offsets = [r * d_samples for r in range(1, R_RETX + 1)]
    lags = [0] + offsets          # 约束 lag：真目标单元(0) + 各假目标位置
    tau_t_samples = int(round(TAU_T_US / baseline.SAMPLE_US))

    print(f"干扰：经典 ISRJ 重复转发 R={R_RETX}，偏移 {[o * baseline.SAMPLE_US for o in offsets]} μs")
    print(f"设计：{len(lags) + 1} 条约束（lag {lags}），M={M_OUTER} 轮交替优化，"
          f"b_min = N·10^(-80/20)")
    params = dataclasses.replace(baseline.params_for(), M=M_OUTER)
    res = joint_design_scene(params, j, offsets, lags, verbose=True)

    x_new, h_new = res["x"], res["h"]
    print(f"\n设计完成：耗时 {res['elapsed_s']:.1f} s")
    print(f"  主瓣约束相对残差      : {res['residual_b_max']:.2e}")
    print(f"  干扰约束相对残差(r=1..R): "
          f"{['%.1e' % v for v in res['residual_b_min']]}")
    print(f"  干扰峰值(r=1..R)      : {['%.2f dB' % v for v in res['il_db']]}")

    # ---- 设计前 / 设计后 ----
    x0 = baseline.random_phase_waveform()
    rk0, y0 = range_profile(x0, x0, j, offsets, tau_t_samples)     # 前：匹配滤波
    rk1, y1 = range_profile(x_new, h_new, j, offsets, tau_t_samples)  # 后：失配滤波

    for tag, rk, ydb in (("前（随机相位+匹配滤波）", rk0, y0),
                         ("后（联合设计+失配滤波）", rk1, y1)):
        vals = []
        for o in offsets:
            p = R_T_KM + o * baseline.SAMPLE_US * KM_PER_US
            near = np.abs(rk - p) <= 0.05
            vals.append(float(ydb[near].max()) if near.any() else float("nan"))
        i_t = int(np.argmin(np.abs(rk - R_T_KM)))
        print(f"\n{tag}：真目标 {ydb[i_t]:+7.2f} dB；"
              f"假目标 {['%+.2f' % v for v in vals]} dB")

    # ---- 绘图 ----
    win = (R_T_KM + WIN_KM[0], R_T_KM + WIN_KM[1])
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True, facecolor=SURFACE)
    for ax, (rk, ydb, title, col) in zip(axes, (
            (rk0, y0, "Before — random-phase waveform + matched filter", "#2a6f97"),
            (rk1, y1, "After — jointly designed waveform + mismatch filter", "#2e7d32"))):
        ax.set_facecolor(SURFACE)
        m = (rk >= win[0]) & (rk <= win[1])
        ax.plot(rk[m], ydb[m], color=col, lw=0.9, zorder=2)
        ax.axvline(R_T_KM, color="#e4572e", lw=0.9, ls="--", zorder=3)
        for k, o in enumerate(offsets):
            p = R_T_KM + o * baseline.SAMPLE_US * KM_PER_US
            ax.axvline(p, color="#c9a227", lw=0.8, ls="--", alpha=0.85, zorder=3)
            near = np.abs(rk - p) <= 0.05
            meas = float(ydb[near].max()) if near.any() else float("nan")
            ax.annotate(f"FT{k + 1}  {meas:+.1f} dB", (p, 4.0), ha="center",
                        fontsize=8.5, color="#8a6d1a",
                        bbox=dict(boxstyle="round,pad=0.15", fc="#fdf6e3",
                                  ec="#c9a227", lw=0.6))
        ax.axhline(0.0, color=TEXT_SECONDARY, lw=0.5, ls=":")
        ax.set_ylim(-100, 8)
        ax.set_ylabel("Normalized level (dB)", color=TEXT_SECONDARY)
        ax.set_title(title, color=TEXT_PRIMARY, fontsize=11)
        ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
        ax.grid(True, color="#e4e2dd", lw=0.5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[1].set_xlim(*win)
    axes[1].set_xlabel("Range (km)", color=TEXT_SECONDARY)
    fig.suptitle(f"ISRJ repeated retransmission jamming (R={R_RETX}, "
                 f"J/S={JSR_DB:g} dB, SNR={SNR_DB:g} dB) — before / after joint design",
                 color=TEXT_PRIMARY, fontsize=12)
    out = OUTDIR / "scene03_before_after_design.png"
    fig.savefig(out, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"\n图已保存：{out}")

    np.savez(OUTDIR / "scene_design_pair.npz", x=x_new, h=h_new, j=j,
             offsets=np.array(offsets))


if __name__ == "__main__":
    main()
