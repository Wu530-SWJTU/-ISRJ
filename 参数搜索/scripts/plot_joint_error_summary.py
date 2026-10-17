"""三对联合误差图合成一张：2 行 × 3 列（上排 3D 曲面、下排等高线）。

数据来源：`run_joint_error_maps.py` 产出的三份 JSON（`结果/联合误差/`）：
    · t0_ts_联合误差.json   （转发时间 × 采样周期）
    · t0_tl_联合误差.json   （转发时间 × 截取长度）
    · tl_ts_联合误差.json   （截取长度 × 采样周期）

统一色标便于横向比较；下排标注设计点（红叉）与"沿另一维 argmin"脊线（橙点），
标题给出形态判定。图内文字一律英文（避免中文字体缺字形）。

口径：搜索线基线 v2（对齐 Zhou 2026）、固定设计对（录波），z = 20lg(|h^H y_j| / |h^H x|)；
按 J/S = 0 dB 幅度口径可直接读作"假目标相对真目标"。

用法：
    .venv/Scripts/python.exe 参数搜索/scripts/plot_joint_error_summary.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUTDIR = ROOT / "参数搜索" / "结果" / "参数调制" / "联合误差"

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

FILES = ["t0_ts_联合误差.json", "t0_tl_联合误差.json", "tl_ts_联合误差.json"]
TITLES = ["(a) $t_0$ × $T_s$", "(b) $t_0$ × $T_L$", "(c) $T_L$ × $T_s$"]
TEXT_PRIMARY, TEXT_SECONDARY = "#0b0b0b", "#52514e"
SURFACE, GRID_C = "#fcfcfb", "#d8d7d2"


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    maps = []
    for fname, ttl in zip(FILES, TITLES):
        d = json.loads((OUTDIR / fname).read_text(encoding="utf-8"))
        g = np.array(d["grid"])
        Z = np.array(d["level_db"])
        X, Y = np.meshgrid(g, g, indexing="ij")
        i0 = int(np.argmin(np.abs(g)))
        maps.append(dict(ttl=ttl, g=g, Z=Z, Xg=X, Yg=Y, i0=i0,
                         xl=d["x_label"], yl=d["y_label"],
                         ridge=g[Z.argmin(axis=1)], s=d["summary"]))
        print(f"  {fname}: 设计点 {d['summary']['design_db']:.2f} dB，"
              f"最小 {d['summary']['min_db']:.2f} dB，{d['summary']['shape']}")

    vmin = min(m["Z"].min() for m in maps)
    vmax = max(m["Z"].max() for m in maps)
    norm = Normalize(vmin=vmin, vmax=vmax)
    levels = np.linspace(vmin, vmax, 19)

    fig = plt.figure(figsize=(17.0, 9.2), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 3, left=0.042, right=0.905, top=0.925, bottom=0.075,
                          hspace=0.26, wspace=0.14)

    for col, m in enumerate(maps):
        ax = fig.add_subplot(gs[0, col], projection="3d")
        ax.plot_surface(m["Xg"], m["Yg"], m["Z"], cmap="viridis", norm=norm,
                        linewidth=0, antialiased=True, alpha=0.95)
        ax.set_xlabel(m["xl"], labelpad=5, fontsize=9)
        ax.set_ylabel(m["yl"], labelpad=5, fontsize=9)
        ax.set_zlabel("Level (dB)", labelpad=3, fontsize=9)
        ax.set_title(m["ttl"], fontsize=10.5, color=TEXT_PRIMARY, pad=2)
        ax.tick_params(labelsize=8, colors=TEXT_SECONDARY)
        ax.view_init(elev=26, azim=-56)

        ax = fig.add_subplot(gs[1, col])
        ax.contourf(m["Xg"], m["Yg"], m["Z"], levels=levels, cmap="viridis", norm=norm)
        cs = ax.contour(m["Xg"], m["Yg"], m["Z"],
                        levels=[l for l in (-70, -60, -50, -40, -30, -25, -20, -15)
                                if vmin < l < vmax],
                        colors="white", linewidths=0.9)
        ax.clabel(cs, fmt="%d dB", fontsize=7.5, colors="white")
        ax.plot(m["g"], m["ridge"], color="#eb6834", linewidth=0, marker="o",
                markersize=3.5, markeredgecolor="white", markeredgewidth=0.5)
        ax.plot(0, 0, marker="x", color="#eb6834", markersize=9, markeredgewidth=2)
        ax.annotate(f"design {m['s']['design_db']:.1f} dB\n"
                    f"min {m['s']['min_db']:.1f} dB @ "
                    f"({m['s']['min_at'][0]:+.2f}, {m['s']['min_at'][1]:+.2f})",
                    xy=(0.03, 0.05), xycoords="axes fraction", fontsize=8.5,
                    color=TEXT_PRIMARY)
        ax.set_xlabel(m["xl"])
        ax.set_ylabel(m["yl"])
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
    fig.axes[1].legend(handles=[proxy_ridge, proxy_design], loc="lower right",
                       frameon=False, fontsize=9, labelcolor=TEXT_PRIMARY)

    cax = fig.add_axes([0.918, 0.12, 0.013, 0.55])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="viridis"), cax=cax)
    cb.set_label("Level after filtering (dB)\n(J/S = 0 dB: false vs true target)",
                 fontsize=9)
    cb.ax.tick_params(labelsize=8, colors=TEXT_SECONDARY)

    out = OUTDIR / "联合误差_三对总览.png"
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"图已保存：{out.name}")


if __name__ == "__main__":
    main()
