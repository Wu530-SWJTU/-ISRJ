"""三对联合误差图合成一张：2 行 × 3 列（上排 3D 曲面、下排等高线）。

数据来源（均由既有脚本产生，本脚本只读取、不重算）：
    · t0_ts_联合误差.json   ← run_2d_error_map.py
    · t0_d_联合误差.json    ← run_joint_error_maps.py
    · d_ts_联合误差.json    ← run_pairwise_maps.py

三列 = 三对参数：t_0×T_s、t_0×d、d×T_s；**统一色标**便于横向比较。
下排标注设计点（红叉）与"沿另一维 argmin"脊线（橙点），标题给出形态判定
（闭合谷 / 仅一维有内部极小 / 两维皆单调；图内文字一律英文，避免字体缺字形）。

口径：固定设计对（t_0 = 2.5 μs、T_s = 10 μs、T_L = 2.5 μs，random_phase、seed=0），
扫实际干扰参数；z = 20lg( |h^H y_j(θ*)| / |h^H x| )，J/S = 0 dB 幅度口径下可直接
读作"假目标相对真目标"。矩阵链未变：J(θ) = diag(j(θ))、H_j = H·diag(j(θ))。

用法：
    .venv/Scripts/python.exe 参数搜索/scripts/plot_joint_error_summary.py
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
from matplotlib.colors import Normalize  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

OUTDIR = ROOT / "参数搜索" / "结果" / "联合误差"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
SURFACE = "#fcfcfb"
GRID_C = "#d8d7d2"

# (文件名, x 路径, y 路径, z 路径, 标题, x 标签, y 标签, x 简称, y 简称)
SPECS = [
    ("t0_ts_联合误差.json", ("axis", "forwarding_start_error_us"),
     ("axis", "sampling_period_error_us"), ("level_db",),
     "(a) $t_0$ × $T_s$", r"$\Delta t_0$ (μs)", r"$\Delta T_s$ (μs)", "$t_0$", "$T_s$"),
    ("t0_d_联合误差.json", ("map_t0_d", "axis_t0"), ("map_t0_d", "axis_d_error"),
     ("map_t0_d", "level_db"),
     "(b) $t_0$ × $d$", r"$\Delta t_0$ (μs)", r"$\Delta d$", "$t_0$", "$d$"),
    ("d_ts_联合误差.json", ("axis_d_error",), ("axis_ts_error_us",), ("level_db",),
     "(c) $d$ × $T_s$", r"$\Delta d$", r"$\Delta T_s$ (μs)", "$d$", "$T_s$"),
]


def _dig(node, path):
    for key in path:
        node = node[key]
    return node


def load(spec):
    fname, px, py, pz, ttl, xl, yl, sx, sy = spec
    data = json.loads((OUTDIR / fname).read_text(encoding="utf-8"))
    X = np.array(_dig(data, px))
    Y = np.array(_dig(data, py))
    Z = np.array(_dig(data, pz))
    Xg, Yg = np.meshgrid(X, Y, indexing="ij")
    i0, j0 = int(np.argmin(np.abs(X))), int(np.argmin(np.abs(Y)))
    mx, px_ = Z[0, j0] - Z[i0, j0], Z[-1, j0] - Z[i0, j0]
    my, py_ = Z[i0, 0] - Z[i0, j0], Z[i0, -1] - Z[i0, j0]
    xi, yi = mx > 0 and px_ > 0, my > 0 and py_ > 0
    if xi and yi:
        shape_zh, shape_en = "闭合谷（两维都有内部极小）", "closed basin (both axes)"
    elif xi:
        shape_zh, shape_en = f"仅 {sx} 维有内部极小", f"interior min only along {sx}"
    elif yi:
        shape_zh, shape_en = f"仅 {sy} 维有内部极小", f"interior min only along {sy}"
    else:
        shape_zh, shape_en = "两维皆单调", "monotone on both axes"
    return dict(fname=fname, X=X, Y=Y, Xg=Xg, Yg=Yg, Z=Z, ttl=ttl, xl=xl, yl=yl,
                i0=i0, j0=j0, shape=shape_zh, shape_en=shape_en,
                min_at=(X[np.unravel_index(Z.argmin(), Z.shape)[0]],
                        Y[np.unravel_index(Z.argmin(), Z.shape)[1]]),
                min_db=float(Z.min()), design_db=float(Z[i0, j0]),
                ridge=Y[Z.argmin(axis=1)])


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    maps = [load(s) for s in SPECS]
    vmin = min(m["min_db"] for m in maps)
    vmax = max(float(m["Z"].max()) for m in maps)
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
        ax.set_title(f"{m['ttl']}  —  {m['shape_en']}", fontsize=10.5,
                     color=TEXT_PRIMARY, pad=2)
        ax.tick_params(labelsize=8, colors=TEXT_SECONDARY)
        ax.view_init(elev=26, azim=-56)

        ax = fig.add_subplot(gs[1, col])
        ax.contourf(m["Xg"], m["Yg"], m["Z"], levels=levels, cmap="viridis", norm=norm)
        cs = ax.contour(m["Xg"], m["Yg"], m["Z"],
                        levels=[-45, -40, -30, -25, -20, -15, -12],
                        colors="white", linewidths=0.9)
        ax.clabel(cs, fmt="%d dB", fontsize=7.5, colors="white")
        ax.plot(m["X"], m["ridge"], color="#eb6834", linewidth=0, marker="o",
                markersize=3.5, markeredgecolor="white", markeredgewidth=0.5)
        ax.plot(0, 0, marker="x", color="#eb6834", markersize=9, markeredgewidth=2)
        ax.annotate(f"design {m['design_db']:.1f} dB\n"
                    f"min {m['min_db']:.1f} dB @ ({m['min_at'][0]:+.2f}, {m['min_at'][1]:+.2f})",
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
    for m in maps:
        print(f"  {m['ttl']:16s} 设计点 {m['design_db']:7.2f} dB ；最小 {m['min_db']:7.2f} dB "
              f"@ ({m['min_at'][0]:+.2f}, {m['min_at'][1]:+.2f}) ；{m['shape']}")


if __name__ == "__main__":
    main()
