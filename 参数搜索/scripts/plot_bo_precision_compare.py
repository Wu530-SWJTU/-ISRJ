# -*- coding: utf-8 -*-
"""BO 精确命中对比：扩V前 vs 扩V后（离线建库 / 在线搜索）。

口径（2026-10-26 用户确认）
    架构   波形离线预训练成库（覆盖 ±5 采样点，1331 份/条件），在线只查表 + 观测；
           一次在线评估 = 一次发射。所以在线成本 = 评估次数（脉冲数），
           单次设计耗时只计入离线建库。
    判据   量化后 θ̂ == θ*（0.02 μs 与精确命中等价，见结果说明）。
数据：`结果/对抗场景/scene08_bo_precision_{before,after}_seed0_budget200.json`

用法
    .venv/Scripts/python.exe 参数搜索/scripts/plot_bo_precision_compare.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUTDIR = ROOT / "参数搜索" / "结果" / "对抗场景"
OUT = OUTDIR / "图" / "scene08_bo_precision_compare.png"

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE, GRID_C = "#fcfcfb", "#d8d7d2"
TEXT_PRIMARY, TEXT_SECONDARY = "#0b0b0b", "#52514e"
BLUE, ORANGE, GRAY = "#2b6cb0", "#eb6834", "#8a8a8a"
CASES = [("before", BLUE, "Before widening"),
         ("after", ORANGE, "After widening")]
N_GRID = 1331


def load(cond: str) -> dict:
    p = OUTDIR / f"scene08_bo_precision_{cond}_seed0_budget200.json"
    return json.loads(p.read_text(encoding="utf-8"))


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    stats = {}
    for cond, _, _ in CASES:
        d = load(cond)
        tr = d["trace"]
        stats[cond] = dict(
            evals=d["exact_hit_eval"], near=d["near_hit_eval"],
            idx=np.array([t["i"] for t in tr], float),
            err=np.array([t["err_samples"] for t in tr], float),
            mean_dt=float(np.mean([t["dt_s"] for t in tr])),
        )
        stats[cond]["offline_h"] = stats[cond]["mean_dt"] * N_GRID / 3600.0

    fig = plt.figure(figsize=(16.0, 4.8), facecolor=SURFACE)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.7, 1.0, 1.0],
                          left=0.05, right=0.985, top=0.84, bottom=0.14,
                          wspace=0.28)

    # (a) 在线搜索过程：best-so-far 误差 vs 评估次数
    ax = fig.add_subplot(gs[0, 0])
    for cond, color, label in CASES:
        s = stats[cond]
        best = np.minimum.accumulate(s["err"])
        ax.plot(s["idx"], s["err"], color=color, linewidth=0.8, alpha=0.25)
        ax.step(s["idx"], best, where="post", color=color, linewidth=2.4,
                label=f"{label} (hit @ {s['evals']})")
        ax.axvline(s["evals"], color=color, linestyle=":", linewidth=1.0, alpha=0.8)
    ax.set_xlabel("Online evaluations (radar pulses)", color=TEXT_PRIMARY)
    ax.set_ylabel("Estimation error (samples, 1 = 0.1 μs)", color=TEXT_PRIMARY)
    ax.set_title("(a) Online search: best-so-far error", fontsize=11,
                 color=TEXT_PRIMARY)
    ax.set_ylim(-0.3, 5.4)
    ax.set_yticks(range(6))
    ax.legend(frameon=True, facecolor=SURFACE, edgecolor=GRID_C, framealpha=0.95,
              fontsize=9, labelcolor=TEXT_PRIMARY, loc="upper right")

    # (b) 在线成本：命中所需脉冲数（含穷举基线）；对数轴用点，不用条形
    ax = fig.add_subplot(gs[0, 1])
    names = ["Exhaustive", "Before", "After"]
    vals = [N_GRID, stats["before"]["evals"], stats["after"]["evals"]]
    cols = [GRAY, BLUE, ORANGE]
    ypos = np.arange(3)[::-1]
    for y, v, c in zip(ypos, vals, cols):
        ax.plot(v, y, "o", color=c, markersize=10, markeredgecolor="white",
                markeredgewidth=0.8)
        ax.text(v * 1.25, y, f"{v}", va="center", fontsize=10, color=TEXT_PRIMARY)
    ax.set_yticks(ypos)
    ax.set_yticklabels(names)
    ax.set_xscale("log")
    ax.set_xlim(15, 4000)
    ax.set_xlabel("Pulses to hit θ* (log scale)", color=TEXT_PRIMARY)
    ax.set_title(f"(b) Online cost  (vs exhaustive: "
                 f"{N_GRID / stats['before']['evals']:.1f}× / "
                 f"{N_GRID / stats['after']['evals']:.1f}×)", fontsize=10.5,
                 color=TEXT_PRIMARY)

    # (c) 离线成本：建库时间（1331 份）
    ax = fig.add_subplot(gs[0, 2])
    names = ["Before", "After"]
    hours = [stats["before"]["offline_h"], stats["after"]["offline_h"]]
    bars = ax.bar(names, hours, color=[BLUE, ORANGE], width=0.55)
    for b, v in zip(bars, hours):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.8, f"{v:.1f} h",
                ha="center", fontsize=9.5, color=TEXT_PRIMARY)
    ax.set_ylabel("Offline build time (h)", color=TEXT_PRIMARY)
    ax.set_ylim(0, max(hours) * 1.22)
    ax.annotate("", xy=(1, hours[1] * 1.02), xytext=(1, hours[0] * 1.02),
                arrowprops=dict(arrowstyle="<->", color=TEXT_SECONDARY, lw=1.0))
    ax.text(1.08, (hours[0] + hours[1]) / 2,
            f"{hours[1] / hours[0]:.1f}×", fontsize=10, color=TEXT_SECONDARY,
            va="center")
    ax.set_title(f"(c) Offline library: {N_GRID} designs", fontsize=11,
                 color=TEXT_PRIMARY)

    for a in fig.axes:
        a.set_facecolor(SURFACE)
        a.grid(True, color=GRID_C, linewidth=0.6, alpha=0.7)
        a.set_axisbelow(True)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
        a.tick_params(labelsize=9, colors=TEXT_SECONDARY)

    fig.suptitle("Waveform-library architecture: offline training vs online search "
                 "(seed 0, b_min = −50 dB, exact-hit criterion)",
                 fontsize=11.5, color=TEXT_PRIMARY, y=0.97)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, facecolor=SURFACE)
    print(f"图已保存：{OUT}")
    print(f"offline: before {stats['before']['offline_h']:.2f} h, "
          f"after {stats['after']['offline_h']:.2f} h")
    plt.close(fig)


if __name__ == "__main__":
    main()
