# -*- coding: utf-8 -*-
"""一维"归一化电平 - 参数误差"曲线（v2 口径，录波口径 C 函数）。

口径（与 `run_joint_error_maps.py` 完全一致，只是把范围从 ±0.5 μs 扩到 ±1 μs）：
    固定设计对（`结果/设计对/joint_design_v2_t0_2us.npz`），**不重新设计**；
    扫实际干扰参数 θ* = θ̂ + Δ，量
        z(Δ) = 20lg( |h^H y_j(θ*)| / |h^H x| )
    矩阵链（零延迟原位）：J(θ) = diag(j(θ))、H_j = H·diag(j(θ))。

误差范围：**±1 μs**（原为 ±0.5 μs，论文 Fig.12 区间），步长 **1 个采样点 = 0.1 μs**（21 档）。
三个参数各扫一维（其余两个置零），画成 1×3 图。

用法：
    .venv/Scripts/python.exe 参数搜索/scripts/run_param_error_1d.py
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
from run_joint_error_maps import DESIGN_NPZ, design_or_load, make_level  # noqa: E402

OUTDIR = ROOT / "参数搜索" / "结果" / "参数调制" / "联合误差"
GRID_US = np.round(np.arange(-1.0, 1.0001, baseline.SAMPLE_US), 4)   # ±1 μs，步长 1 采样点
OLD_RANGE = 0.5                                                     # 原范围（图上标出）
TEXT_PRIMARY, TEXT_SECONDARY = "#0b0b0b", "#52514e"
SURFACE = "#fcfcfb"
SERIES = [("$\\Delta t_0$", "#1f4e79", lambda L, d: L(e_t0=d)),
          ("$\\Delta T_L$", "#2e7d32", lambda L, d: L(e_tl=d)),
          ("$\\Delta T_s$", "#c0392b", lambda L, d: L(e_ts=d))]


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    OUTDIR.mkdir(parents=True, exist_ok=True)

    print("=== 一维参数误差曲线（v2 口径，录波口径 C 函数）===")
    print(f"  误差范围 ±{GRID_US.max():g} μs，步长 {baseline.SAMPLE_US:g} μs"
          f"（{GRID_US.size} 档）")
    x, h = design_or_load()
    level = make_level(x, h)

    curves = {}
    for name, _c, f in SERIES:
        curves[name] = np.array([f(level, float(d)) for d in GRID_US])

    print(f"\n{'Δ(μs)':>7} " + " ".join(f"{n:>10}" for n, _, _ in SERIES))
    for i, d in enumerate(GRID_US):
        if abs(d) <= 1e-9 or abs(abs(d) - OLD_RANGE) < 1e-9 or abs(abs(d) - 1.0) < 1e-9:
            print(f"{d:>7.1f} " + " ".join(f"{curves[n][i]:>10.2f}"
                                           for n, _, _ in SERIES))

    # 谷底与半宽（相对谷底的抬升量）
    stats = {}
    for n, _, _ in SERIES:
        v = curves[n]
        i0 = int(np.argmin(np.abs(GRID_US)))
        stats[n] = {"design_db": float(v[i0]), "min_db": float(v.min()),
                    "min_at_us": float(GRID_US[int(v.argmin())]),
                    "at_pm_0.5": [float(v[i0 - 5]), float(v[i0 + 5])],
                    "at_pm_1.0": [float(v[i0 - 10]), float(v[i0 + 10])]}
        s = stats[n]
        print(f"  {n:>10}: 谷底 {s['design_db']:.2f} dB @ {s['min_at_us']:+.1f} μs ；"
              f"±0.5 μs 抬升 {s['at_pm_0.5'][1] - s['design_db']:+.1f} / "
              f"{s['at_pm_0.5'][0] - s['design_db']:+.1f} dB ；"
              f"±1.0 μs 抬升 {s['at_pm_1.0'][1] - s['design_db']:+.1f} / "
              f"{s['at_pm_1.0'][0] - s['design_db']:+.1f} dB")

    # ---- 绘图 ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True, facecolor=SURFACE)
    for ax, (name, color, _f) in zip(axes, SERIES):
        ax.set_facecolor(SURFACE)
        ax.axvspan(-OLD_RANGE, OLD_RANGE, color="#eef3f8", zorder=1)
        ax.plot(GRID_US, curves[name], color=color, lw=1.8, marker="o", ms=3.2,
                zorder=3)
        ax.axvline(0, color="#b8b6b0", lw=0.8, ls=":", zorder=2)
        ax.scatter([0], [stats[name]["design_db"]], color=color, s=45, zorder=4,
                   edgecolor="white", linewidth=0.8)
        ax.annotate(f"{stats[name]['design_db']:.1f} dB",
                    (0, stats[name]["design_db"]), xytext=(0, -14),
                    textcoords="offset points", ha="center", va="top",
                    fontsize=8.5, color=color)
        ax.set_xlabel(f"{name} error (μs)", color=TEXT_SECONDARY, fontsize=9.5)
        ax.set_title(f"Normalized level vs {name}", color=TEXT_PRIMARY, fontsize=10.5)
        ax.set_xlim(-1.05, 1.05)
        ax.set_xticks(np.arange(-1.0, 1.01, 0.5))
        ax.tick_params(colors=TEXT_SECONDARY, labelsize=8.5)
        ax.grid(True, color="#e4e2dd", lw=0.4)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("Normalized level (dB)", color=TEXT_SECONDARY, fontsize=9.5)
    axes[0].text(-OLD_RANGE + 0.02, axes[0].get_ylim()[0] + 2,
                 "shaded: original ±0.5 μs", fontsize=7.5, color="#8a9aa8")
    fig.suptitle("1-D parameter error curves — fixed design pair, v2 caliber "
                 "(sweep range extended to ±1 μs)", color=TEXT_PRIMARY, fontsize=12)
    out = OUTDIR / "一维参数误差曲线.png"
    fig.savefig(out, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)

    (OUTDIR / "一维参数误差曲线.json").write_text(json.dumps({
        "grid_us": GRID_US.tolist(), "old_range_us": OLD_RANGE,
        "curves_db": {n: curves[n].tolist() for n, _, _ in SERIES},
        "stats": stats,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n图已保存：{out}")
    print(f"数据已保存：{OUTDIR / '一维参数误差曲线.json'}")


if __name__ == "__main__":
    main()
