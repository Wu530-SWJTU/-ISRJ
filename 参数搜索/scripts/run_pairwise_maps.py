"""三对参数联合误差总览：t_0×T_s、t_0×d（复用）+ d×T_s（本次计算）。

设计对固定（t_0 = 2.5 μs、T_s = 10 μs、T_L = 2.5 μs、random_phase、seed=0），
扫**实际**干扰参数（固定对/录波口径）：

    y_j(θ*) = x ⊙ j(θ*)，   z = 20lg( |h^H y_j(θ*)| / |h^H x| )

矩阵链（未变）：J(θ) = diag(j(θ))、H_j = H·diag(j(θ))、X_j 由 x⊙j 铺成。

网格（三对一致，各 21×21）：
    Δt_0 ∈ ±2.5 μs（步长 0.25）、ΔT_s ∈ ±2.5 μs（步长 0.25）、Δd ∈ ±0.20（步长 0.02）
合法性：窗口不跨周期 `t_0 + T_L ≤ T_s`；最坏点 (Δt_0=+2.5, Δd=+0.20, ΔT_s=-2.5)
        → 5.0 + 4.5 = 9.5 ≤ 7.5？否 —— 见下
        实际最坏组合出现在 d×T_s 图中：(t_0=2.5, T_L=4.5, T_s=7.5) → 7.0 ≤ 7.5 ✓
        t_0×d 图中 Δt_0 与 Δd 同时取正时会越界，该角落在前一轮图中已标注、此处沿用。

用法：
    .venv/Scripts/python.exe 参数搜索/scripts/run_pairwise_maps.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from isrj_joint_design.matrices import cross_correlation  # noqa: E402
from isrj_joint_design.params import GlobalParams, generate_isrj_sampling_function  # noqa: E402

N = 800
T0_US, TS_US, TAU_US = 2.5, 10.0, 2.5
D_DESIGN = TAU_US / TS_US
GRID_T0 = np.arange(-2.5, 2.5001, 0.25)
GRID_TS = np.arange(-2.5, 2.5001, 0.25)
GRID_D = np.arange(-0.20, 0.2001, 0.02)
DESIGN_NPZ = ROOT / "参数搜索" / "结果" / "窗口起点" / "joint_design_result_t0_2.5us.npz"
MAP_T0TS = ROOT / "参数搜索" / "结果" / "联合误差" / "t0_ts_联合误差.json"
MAP_T0D = ROOT / "参数搜索" / "结果" / "联合误差" / "t0_d_联合误差.json"
OUTDIR = ROOT / "参数搜索" / "结果" / "联合误差"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
SURFACE = "#fcfcfb"
GRID_C = "#d8d7d2"


def level_db(x, h, e_t0, e_ts, dd):
    p = GlobalParams(N=N, tau=(D_DESIGN + dd) * TS_US * 1e-6,
                     T_s=(TS_US + e_ts) * 1e-6,
                     tau_offset=(T0_US + e_t0) * 1e-6, seed=0)
    r = cross_correlation(x * generate_isrj_sampling_function(p), h)
    return float(20 * np.log10(abs(r[N - 1]) / abs(np.vdot(h, x)) + 1e-30))


def verdict(Z, X, Y, i0, j0, name, ttl_x, ttl_y):
    """Z[i, j]：i 沿 X、j 沿 Y。端点为固定另一维为 0 时走到该维两端的抬升。"""
    minus_x = float(Z[0, j0] - Z[i0, j0])
    plus_x = float(Z[-1, j0] - Z[i0, j0])
    minus_y = float(Z[i0, 0] - Z[i0, j0])
    plus_y = float(Z[i0, -1] - Z[i0, j0])
    x_interior = minus_x > 0 and plus_x > 0
    y_interior = minus_y > 0 and plus_y > 0
    if x_interior and y_interior:
        shape = "闭合谷（两维都有内部极小）"
    elif x_interior:
        shape = f"仅 {ttl_x} 维有内部极小（{ttl_y} 维单调）"
    elif y_interior:
        shape = f"仅 {ttl_y} 维有内部极小（{ttl_x} 维单调）"
    else:
        shape = "两维皆单调"
    gmin = np.unravel_index(Z.argmin(), Z.shape)
    return {
        "map": name, "x": ttl_x, "y": ttl_y,
        "design_db": float(Z[i0, j0]), "min_db": float(Z.min()),
        "min_at": [float(X[gmin[0]]), float(Y[gmin[1]])],
        "x_minus": minus_x, "x_plus": plus_x, "y_minus": minus_y, "y_plus": plus_y,
        "x_interior": bool(x_interior), "y_interior": bool(y_interior),
        "shape": shape, "closed_basin": bool(x_interior and y_interior),
        "ridge_x": np.asarray(X).tolist(),
        "ridge_y": np.asarray(Y)[Z.argmin(axis=1)].tolist(),
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    OUTDIR.mkdir(parents=True, exist_ok=True)

    d = np.load(DESIGN_NPZ)
    x, h = d["x"], d["h"]
    print("=== 三对参数联合误差总览 ===")

    a = json.loads(MAP_T0TS.read_text(encoding="utf-8"))
    XA = np.array(a["axis"]["forwarding_start_error_us"])
    YA = np.array(a["axis"]["sampling_period_error_us"])
    ZA = np.array(a["level_db"])
    print(f"  t_0 × T_s：复用 {MAP_T0TS.name}")

    b = json.loads(MAP_T0D.read_text(encoding="utf-8"))
    XB = np.array(b["map_t0_d"]["axis_t0"])
    YB = np.array(b["map_t0_d"]["axis_d_error"])
    ZB = np.array(b["map_t0_d"]["level_db"])
    print(f"  t_0 × d  ：复用 {MAP_T0D.name}")

    ZC = np.array([[level_db(x, h, 0.0, ets, dd) for ets in GRID_TS] for dd in GRID_D])
    print(f"  d × T_s  ：本次计算 {ZC.shape[0]}×{ZC.shape[1]} = {ZC.size} 点")

    i_a = (int(np.argmin(np.abs(XA))), int(np.argmin(np.abs(YA))))
    i_b = (int(np.argmin(np.abs(XB))), int(np.argmin(np.abs(YB))))
    i_c = (int(np.argmin(np.abs(GRID_D))), int(np.argmin(np.abs(GRID_TS))))
    rows = [
        verdict(ZA, XA, YA, *i_a, "t_0 × T_s", "Δt_0 (μs)", "ΔT_s (μs)"),
        verdict(ZB, XB, YB, *i_b, "t_0 × d", "Δt_0 (μs)", "Δd"),
        verdict(ZC, GRID_D, GRID_TS, i_c[0], i_c[1], "d × T_s", "Δd", "ΔT_s (μs)"),
    ]
    for r in rows:
        print(f"  {r['map']:10s} 设计点 {r['design_db']:7.2f} dB，全图最小 {r['min_db']:7.2f} dB "
              f"@ ({r['min_at'][0]:+.2f}, {r['min_at'][1]:+.2f})")
        print(f"             x 端 −/{r['x_minus']:+.2f} ＋/{r['x_plus']:+.2f} dB ；"
              f"y 端 −/{r['y_minus']:+.2f} ＋/{r['y_plus']:+.2f} dB"
              f" ⇒ {r['shape']}")

    proxy = Line2D([], [], color="#eb6834", marker="o", linestyle="none",
                   markeredgecolor="white", markeredgewidth=0.6, markersize=4,
                   label="argmin over the other axis")
    proxy_x = Line2D([], [], color="#eb6834", marker="x", linestyle="none",
                     markersize=9, markeredgewidth=2, label="design point")

    fig, axes = plt.subplots(1, 3, figsize=(18.5, 5.4), facecolor=SURFACE)
    for ax, Z, X, Y, r in zip(axes, (ZA, ZB, ZC), (XA, XB, GRID_D),
                              (YA, YB, GRID_TS), rows):
        fn = ("closed basin" if r["closed_basin"]
              else ("interior min on one axis only" if (r["x_interior"] or r["y_interior"])
                    else "monotone on both axes"))
        ax.contourf(np.meshgrid(X, Y, indexing="ij")[0], np.meshgrid(X, Y, indexing="ij")[1],
                    Z, levels=18, cmap="viridis")
        cs = ax.contour(np.meshgrid(X, Y, indexing="ij")[0], np.meshgrid(X, Y, indexing="ij")[1],
                        Z, levels=[-25, -20, -15, -12, -10], colors="white", linewidths=0.9)
        ax.clabel(cs, fmt="%d", fontsize=8, colors="white")
        ax.plot(r["ridge_x"], r["ridge_y"], color="#eb6834", linewidth=0, marker="o",
                markersize=4, markeredgecolor="white", markeredgewidth=0.6)
        ax.plot(0, 0, marker="x", color="#eb6834", markersize=9, markeredgewidth=2)
        ax.set_xlabel(f"{r['x']}")
        ax.set_ylabel(f"{r['y']}")
        ax.set_title(f"{r['map']}   ({fn})", fontsize=11, color=TEXT_PRIMARY)
        ax.set_facecolor(SURFACE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.grid(True, color=GRID_C, linewidth=0.6, alpha=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=9, colors=TEXT_SECONDARY)
    axes[0].legend(handles=[proxy, proxy_x], loc="upper left", frameon=False,
                   fontsize=9, labelcolor=TEXT_PRIMARY)
    fig.tight_layout()
    out = OUTDIR / "三对联合误差_总览.png"
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"  图已保存：{out.name}")

    (OUTDIR / "d_ts_联合误差.json").write_text(json.dumps({
        "design": {"npz": DESIGN_NPZ.name, "t0_us": T0_US, "ts_us": TS_US,
                   "tau_us": TAU_US, "d": D_DESIGN, "N": N, "seed": 0},
        "axis_d_error": GRID_D.tolist(), "axis_ts_error_us": GRID_TS.tolist(),
        "level_db": ZC.tolist(), "summary": rows,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = ["# 三对参数联合误差总览", "",
             f"固定设计对 `{DESIGN_NPZ.name}`（t_0 = 2.5 μs、T_s = 10 μs、T_L = 2.5 μs）；"
             f"网格 21×21。", "",
             "| 参数对 | 设计点 (dB) | 全图最小 (dB) | 最小值位置 | x 端 (−/＋) | y 端 (−/＋) | 形态 |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        shape = r["shape"]
        lines.append(f"| {r['map']} | {r['design_db']:.2f} | {r['min_db']:.2f} | "
                     f"({r['min_at'][0]:+.2f}, {r['min_at'][1]:+.2f}) | "
                     f"{r['x_minus']:+.2f} / {r['x_plus']:+.2f} | "
                     f"{r['y_minus']:+.2f} / {r['y_plus']:+.2f} | {shape} |")
    lines += ["", "（端点为固定另一维为 0 时，走到该维两端相对设计点的抬升，dB；负值表示比设计点更好。）"]
    (OUTDIR / "三对联合_结果表.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("  JSON 与结果表已保存")


if __name__ == "__main__":
    main()
