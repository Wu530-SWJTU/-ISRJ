"""纵坐标口径对比：同一组 t_0 扫描数据，三种画法并排。

用来回答"不做归一化会不会不一样"：
    ① 归一化 dB（现行）：20lg( |h^H(x⊙j)| / |h^H x| )，分母 = b_max = 800（常数）
    ② 绝对 dB（不做归一化）：20lg|h^H(x⊙j)| = ① + 20lg(800) = ① + 58.06 dB
       —— 只是上下平移，形状不变（分母是常数，除以常数不改形状）
    ③ 线性比值（不取 dB）：|h^H(x⊙j)| / |h^H x|
       —— 同一组数，视觉上陡得多：0.032 → 0.34 看起来像"从零陡升"

结论：形状由被测量的物理量决定，不由归一化决定；只有"取不取 dB"改变观感，
       真正改变形状要换指标（阈值/检测型）或换误差维度。

用法：
    .venv/Scripts/python.exe 参数搜索/scripts/plot_scale_compare.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUTDIR = ROOT / "参数搜索" / "结果" / "窗口起点"
SERIES_1 = "#2a78d6"
SERIES_2 = "#eb6834"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
SURFACE = "#fcfcfb"
GRID = "#d8d7d2"
B_MAX = 800.0            # |h^H x| = b_max = N（设计约束钉死）


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    ax.xaxis.label.set_color(TEXT_SECONDARY)
    ax.yaxis.label.set_color(TEXT_SECONDARY)
    ax.title.set_color(TEXT_PRIMARY)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    data = json.loads((OUTDIR / "t0_sweep.json").read_text(encoding="utf-8"))
    err = np.array(data["errors_us"])
    zero_db = np.array(data["zero_delay_db"])          # ① 归一化 dB
    ratio = 10 ** (zero_db / 20)                       # ③ 线性比值
    abs_db = zero_db + 20 * np.log10(B_MAX)            # ② 绝对 dB

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.6), facecolor=SURFACE)

    ax = axes[0]
    ax.plot(err, zero_db, color=SERIES_1, linewidth=2, marker="o", markersize=4)
    ax.axhline(-30, color=TEXT_SECONDARY, linewidth=1.0, linestyle="-.")
    ax.axvline(0.0, color=TEXT_SECONDARY, linewidth=1.0, linestyle="--", alpha=0.8)
    ax.set_title("(a) Normalized level (dB)  — current", fontsize=11)
    ax.set_ylabel("20lg( |h^H(y_j)| / |h^H x| )  (dB)")
    _style(ax)

    ax = axes[1]
    ax.plot(err, abs_db, color=SERIES_2, linewidth=2, marker="o", markersize=4)
    ax.axvline(0.0, color=TEXT_SECONDARY, linewidth=1.0, linestyle="--", alpha=0.8)
    ax.set_title("(b) Absolute level (dB)  — not normalized", fontsize=11)
    ax.set_ylabel("20lg|h^H(y_j)|  (dB)")
    ax.annotate("= curve (a) shifted by +58.06 dB\ndenominator is a constant -> same shape",
                xy=(0.03, 0.86), xycoords="axes fraction",
                fontsize=9, color=TEXT_PRIMARY)
    _style(ax)

    ax = axes[2]
    ax.plot(err, ratio, color=SERIES_1, linewidth=2, marker="o", markersize=4)
    ax.axvline(0.0, color=TEXT_SECONDARY, linewidth=1.0, linestyle="--", alpha=0.8)
    ax.set_ylim(0, 0.36)
    ax.set_title("(c) Linear ratio  — same data, no dB", fontsize=11)
    ax.set_ylabel("|h^H(y_j)| / |h^H x|")
    ax.annotate("same numbers, visually much steeper:\n0.032 -> 0.336 (10.7x)",
                xy=(0.03, 0.86), xycoords="axes fraction",
                fontsize=9, color=TEXT_PRIMARY)
    _style(ax)

    for ax in axes:
        ax.set_xlabel("Window-start error (μs)")

    fig.tight_layout()
    out = OUTDIR / "对比_dB与线性刻度.png"
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"图已保存：{out.name}")
    print()
    print("误差(μs)   ①归一化dB   ②绝对dB(=①+58.06)   ③线性比值   ③相对谷底的倍数")
    for i in range(0, len(err), 10):
        print(f"{err[i]:+7.2f}   {zero_db[i]:9.2f}   {abs_db[i]:16.2f}   {ratio[i]:9.4f}"
              f"   {ratio[i]/ratio[np.argmin(abs(err))]:8.2f}×")
    i0 = int(np.argmin(abs(err)))
    print(f"{err[i0]:+7.2f}   {zero_db[i0]:9.2f}   {abs_db[i0]:16.2f}   {ratio[i0]:9.4f}   1.00×")


if __name__ == "__main__":
    main()
