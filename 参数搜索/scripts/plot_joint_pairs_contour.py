# -*- coding: utf-8 -*-
"""三对联合误差总览：4 行 × 3 列（上排 3D 曲面、下排等高线，扩V前 / 扩V后各两行）。

版式对齐 `结果/参数调制/联合误差/联合误差_三对总览.png`：统一 viridis 色标、
3D 视角一致、下排标设计点（橙叉）与沿另一维的 argmin 脊线（橙点）。

数据：`结果/对抗场景/scene08_joint_pairs_contour.json`
（3 对 × 扩V前后 × 11×11；网格为误差采样点，1 点 = 0.1 μs，范围 ±0.5 μs）
数组方向：Z[i, j] 中 j 是第一个参数（横轴），i 是第二个参数（纵轴）。

用法
    .venv/Scripts/python.exe 参数搜索/scripts/plot_joint_pairs_contour.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUTDIR = ROOT / "参数搜索" / "结果" / "对抗场景"
SRC = OUTDIR / "scene08_joint_pairs_contour.json"
OUT = OUTDIR / "图" / "scene08_joint_pairs_overview.png"

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

SAMPLE_US = 0.1
PAIRS = [("t_0_T_s", r"$t_0$ × $T_s$", r"$\Delta t_0$", r"$\Delta T_s$"),
         ("t_0_T_L", r"$t_0$ × $T_L$", r"$\Delta t_0$", r"$\Delta T_L$"),
         ("T_L_T_s", r"$T_L$ × $T_s$", r"$\Delta T_L$", r"$\Delta T_s$")]
STATES = [("before", "Before widening"), ("after", "After widening")]
TEXT_PRIMARY, TEXT_SECONDARY = "#0b0b0b", "#52514e"
SURFACE, GRID_C = "#fcfcfb", "#d8d7d2"


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    d = json.loads(SRC.read_text(encoding="utf-8"))
    g = np.arange(-5, 6) * SAMPLE_US
    X, Y = np.meshgrid(g, g)

    maps = []
    for state, label in STATES:
        for key, ttl, xl, yl in PAIRS:
            maps.append(dict(z=np.array(d[key][state], dtype=float), ttl=ttl,
                             xl=xl, yl=yl, state=label))

    vmin = min(m["z"].min() for m in maps)
    vmax = max(m["z"].max() for m in maps)
    norm = Normalize(vmin=vmin, vmax=vmax)
    levels = np.linspace(vmin, vmax, 19)
    line_levels = [l for l in (-45, -40, -35, -30, -25, -20, -15) if vmin < l < vmax]

    fig = plt.figure(figsize=(17.0, 17.2), facecolor=SURFACE)
    gs = fig.add_gridspec(4, 3, left=0.045, right=0.905, top=0.955, bottom=0.05,
                          hspace=0.24, wspace=0.16)

    letters = "abcdef"
    first_contour = None
    for row_pair in range(2):
        for col in range(3):
            m = maps[row_pair * 3 + col]
            r = row_pair * 2
            idx = row_pair * 3 + col

            ax = fig.add_subplot(gs[r, col], projection="3d")
            ax.plot_surface(X, Y, m["z"], cmap="viridis", norm=norm,
                            linewidth=0, antialiased=True, alpha=0.95)
            ax.set_xlabel(m["xl"] + " (μs)", labelpad=8, fontsize=9)
            ax.set_ylabel(m["yl"] + " (μs)", labelpad=13, fontsize=9)
            ax.set_zlabel("Level (dB)", labelpad=3, fontsize=9)
            ax.set_zlim(vmin, vmax)
            ax.set_title(f"({letters[idx]}) {m['ttl']} · {m['state'].split()[0].lower()}",
                         fontsize=10.5, color=TEXT_PRIMARY, pad=2)
            ax.tick_params(labelsize=8, colors=TEXT_SECONDARY)
            ax.view_init(elev=26, azim=-56)

            ax = fig.add_subplot(gs[r + 1, col])
            if first_contour is None:
                first_contour = ax
            ax.contourf(X, Y, m["z"], levels=levels, cmap="viridis", norm=norm)
            cs = ax.contour(X, Y, m["z"], levels=line_levels,
                            colors="white", linewidths=0.9)
            ax.clabel(cs, fmt="%d dB", fontsize=7.5, colors="white")

            # 沿另一维的 argmin（横轴每一点对应的纵轴最优值）
            i_arg = m["z"].argmin(axis=0)
            ax.plot(g, g[i_arg], color="#eb6834", linewidth=0, marker="o",
                    markersize=3.5, markeredgecolor="white", markeredgewidth=0.5)
            ax.plot(0, 0, marker="x", color="#eb6834", markersize=9, markeredgewidth=2)

            z = m["z"]
            i0, j0 = np.unravel_index(z.argmin(), z.shape)
            ax.annotate(f"{m['state']}\ndesign {z[5, 5]:.1f} dB\n"
                        f"min {z[i0, j0]:.1f} dB @ ({g[j0]:+.2f}, {g[i0]:+.2f})",
                        xy=(0.03, 0.04), xycoords="axes fraction", fontsize=8.5,
                        color=TEXT_PRIMARY, va="bottom",
                        bbox=dict(boxstyle="round,pad=0.3", facecolor=SURFACE,
                                  edgecolor="none", alpha=0.9))
            ax.set_xlabel(m["xl"] + " (μs)")
            ax.set_ylabel(m["yl"] + " (μs)")
            ax.set_xticks(np.arange(-0.4, 0.41, 0.2))
            ax.set_yticks(np.arange(-0.4, 0.41, 0.2))
            ax.set_facecolor(SURFACE)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            ax.grid(True, color=GRID_C, linewidth=0.6, alpha=0.6)
            ax.set_axisbelow(True)
            ax.tick_params(labelsize=9, colors=TEXT_SECONDARY)

    proxy_ridge = Line2D([], [], color="#eb6834", marker="o", linestyle="none",
                         markeredgecolor="white", markeredgewidth=0.5, markersize=3.5,
                         label="argmin over the other axis")
    proxy_design = Line2D([], [], color="#eb6834", marker="x", linestyle="none",
                          markersize=9, markeredgewidth=2, label="design point")
    first_contour.legend(handles=[proxy_ridge, proxy_design], loc="lower right",
                         frameon=False, fontsize=9, labelcolor=TEXT_PRIMARY)

    cax = fig.add_axes([0.918, 0.08, 0.013, 0.84])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="viridis"), cax=cax)
    cb.set_label("Worst false-target level (dB)\n(relative to true target)",
                 fontsize=9)
    cb.ax.tick_params(labelsize=8, colors=TEXT_SECONDARY)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, facecolor=SURFACE)
    print(f"图已保存：{OUT}")
    plt.close(fig)


if __name__ == "__main__":
    main()
