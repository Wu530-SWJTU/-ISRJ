"""联合误差图：转发起始时间 t_0 × 另两个参数（T_s、d）。

两两联合：
    (a) t_0 × T_s —— 复用已有结果（`结果/二维误差图/二维误差图.json`）
    (b) t_0 × d   —— 本次计算（d 即占空比，固定 T_s = 10 μs 时等价于截取长度 T_L）

口径：固定设计对（t_0 = 2.5 μs、T_s = 10 μs、T_L = 2.5 μs，random_phase、seed=0），
扫**实际**干扰参数（录波口径）：

    y_j(θ*) = x ⊙ j(θ*)，   z = 20lg( |h^H y_j(θ*)| / |h^H x| )

矩阵链（未变）：J(θ) = diag(j(θ))、H_j = H·diag(j(θ))、X_j 由 x⊙j 铺成。

网格：Δt_0 ∈ ±2.5 μs（步长 0.25，21 点）；ΔT_s ∈ ±2.5 μs / Δd ∈ ±0.20（各 21 点）。
合法性：窗口不跨周期 `t_0 + T_L ≤ T_s`；本网格最坏点 (t_0 = 5.0, T_L = 4.5) → 9.5 ≤ 10 ✓。

用法：
    .venv/Scripts/python.exe 参数搜索/scripts/run_joint_error_maps.py
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

from isrj_joint_design.matrices import cross_correlation  # noqa: E402
from isrj_joint_design.params import GlobalParams, generate_isrj_sampling_function  # noqa: E402

N = 800
T0_US, TS_US, TAU_US = 2.5, 10.0, 2.5
D_DESIGN = TAU_US / TS_US
GRID_T0 = np.arange(-2.5, 2.5001, 0.25)
GRID_D = np.arange(-0.20, 0.2001, 0.02)
DESIGN_NPZ = ROOT / "参数搜索" / "结果" / "窗口起点" / "joint_design_result_t0_2.5us.npz"
MAP_TS_JSON = ROOT / "参数搜索" / "结果" / "二维误差图" / "二维误差图.json"
OUTDIR = ROOT / "参数搜索" / "结果" / "联合误差"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
SURFACE = "#fcfcfb"
GRID_C = "#d8d7d2"


def level_db(x: np.ndarray, h: np.ndarray, e_t0: float, d: float) -> float:
    p = GlobalParams(N=N, tau=d * TS_US * 1e-6, T_s=TS_US * 1e-6,
                     tau_offset=(T0_US + e_t0) * 1e-6, seed=0)
    r = cross_correlation(x * generate_isrj_sampling_function(p), h)
    return float(20 * np.log10(abs(r[N - 1]) / abs(np.vdot(h, x)) + 1e-30))


def _style2d(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(True, color=GRID_C, linewidth=0.6, alpha=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=9, colors=TEXT_SECONDARY)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    OUTDIR.mkdir(parents=True, exist_ok=True)

    d = np.load(DESIGN_NPZ)
    x, h = d["x"], d["h"]
    print("=== 联合误差图：t_0 × T_s 与 t_0 × d ===")

    # (a) 复用已有的 t_0 × T_s 结果
    ts_data = json.loads(MAP_TS_JSON.read_text(encoding="utf-8"))
    XA = np.array(ts_data["axis"]["forwarding_start_error_us"])
    YA = np.array(ts_data["axis"]["sampling_period_error_us"])
    ZA = np.array(ts_data["level_db"])
    print(f"  (a) t_0 × T_s：复用 {MAP_TS_JSON.name}（{ZA.shape[0]}×{ZA.shape[1]}）")

    # (b) 计算 t_0 × d
    ZB = np.array([[level_db(x, h, e0, D_DESIGN + dd) for dd in GRID_D] for e0 in GRID_T0])
    print(f"  (b) t_0 × d：{ZB.shape[0]}×{ZB.shape[1]} = {ZB.size} 点（本次计算）")

    def summarize(Z, X, Y, name):
        i0 = int(np.argmin(np.abs(X)))
        j0 = int(np.argmin(np.abs(Y)))
        gmin = np.unravel_index(Z.argmin(), Z.shape)
        ridge = Y[Z.argmin(axis=1)]              # 每个 t_0 误差下，另一维的最优值
        spread = float(np.max(ridge) - np.min(ridge))
        return {
            "map": name, "design_point_db": float(Z[i0, j0]),
            "min_db": float(Z.min()), "min_at": [float(X[gmin[0]]), float(Y[gmin[1]])],
            "second_axis_at_design": float(Z[i0, j0]),
            "second_axis_best_in_column": float(Z[:, j0].min()),
            "t0_sensitivity_0p5": float(Z[i0 + 2, j0] - Z[i0, j0])
            if i0 + 2 < len(X) else float("nan"),
            "second_axis_at_plus": float(Z[i0, -1] - Z[i0, j0]),
            "second_axis_at_minus": float(Z[i0, 0] - Z[i0, j0]),
            "ridge_spread": spread,
        }

    rows = [summarize(ZA, XA, YA, "t_0 × T_s"), summarize(ZB, GRID_T0, GRID_D, "t_0 × d")]
    for r in rows:
        print(f"  {r['map']}：设计点 {r['design_point_db']:.2f} dB，"
              f"全图最小 {r['min_db']:.2f} dB @ (Δt_0 = {r['min_at'][0]:+.2f}, "
              f"另一维 {r['min_at'][1]:+.2f})")
        print(f"    t_0 方向偏 ±0.5 μs：{r['t0_sensitivity_0p5']:+.2f} dB ；"
              f"另一维偏 −/＋端：{r['second_axis_at_minus']:+.2f} / {r['second_axis_at_plus']:+.2f} dB"
              f" ⇒ {'有内部极小' if r['second_axis_at_minus'] > 0 and r['second_axis_at_plus'] > 0 else '单调（极小在边界）'}")

    from matplotlib.lines import Line2D
    proxy_ridge = Line2D([], [], color="#eb6834", marker="o", linestyle="none",
                         markeredgecolor="white", markeredgewidth=0.6, markersize=4,
                         label="argmin over the other axis")

    fig = plt.figure(figsize=(15.5, 10.5), facecolor=SURFACE)
    Xa, Ya = np.meshgrid(XA, YA, indexing="ij")
    Xb, Yb = np.meshgrid(GRID_T0, GRID_D, indexing="ij")

    for col, (X, Y, Z, xa, ya, ttl, ylab) in enumerate((
            (Xa, Ya, ZA, XA, YA, "(a) t_0 × T_s", "Sampling-period error (μs)"),
            (Xb, Yb, ZB, GRID_T0, GRID_D, "(b) t_0 × d", "Duty-cycle error Δd"))):
        ax = fig.add_subplot(2, 2, col + 1, projection="3d")
        surf = ax.plot_surface(X, Y, Z, cmap="viridis", linewidth=0, antialiased=True,
                               alpha=0.95)
        ax.set_xlabel("Forwarding-start error (μs)", labelpad=8, fontsize=9)
        ax.set_ylabel(ylab, labelpad=8, fontsize=9)
        ax.set_zlabel("Level (dB)", labelpad=6, fontsize=9)
        ax.set_title(ttl, fontsize=11, color=TEXT_PRIMARY)
        ax.tick_params(labelsize=8, colors=TEXT_SECONDARY)
        ax.view_init(elev=26, azim=-56)
        fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.08, label="level (dB)")

        ax = fig.add_subplot(2, 2, col + 3)
        cf = ax.contourf(X, Y, Z, levels=18, cmap="viridis")
        cs = ax.contour(X, Y, Z, levels=[-25, -20, -15, -12, -10],
                        colors="white", linewidths=0.9)
        ax.clabel(cs, fmt="%d dB", fontsize=8, colors="white")
        ax.plot(np.asarray(xa), np.asarray(ya)[Z.argmin(axis=1)], color="#eb6834",
                linewidth=0, marker="o", markersize=4, markeredgecolor="white",
                markeredgewidth=0.6)
        ax.plot(0, 0, marker="x", color="#eb6834", markersize=10, markeredgewidth=2)
        ax.set_xlabel("Forwarding-start error (μs)")
        ax.set_ylabel(ylab)
        ax.set_title(ttl + " (contours)", fontsize=11, color=TEXT_PRIMARY)
        ax.legend(handles=[proxy_ridge], loc="upper left", frameon=False,
                  fontsize=9, labelcolor=TEXT_PRIMARY)
        _style2d(ax)
        fig.colorbar(cf, ax=ax, shrink=0.9, pad=0.03, label="level (dB)")

    fig.tight_layout()
    out = OUTDIR / "联合误差_两两对比.png"
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"  图已保存：{out.name}")

    (OUTDIR / "联合误差.json").write_text(json.dumps({
        "design": {"npz": DESIGN_NPZ.name, "t0_us": T0_US, "ts_us": TS_US,
                   "tau_us": TAU_US, "d": D_DESIGN, "N": N, "seed": 0},
        "map_t0_ts": {"source": MAP_TS_JSON.name, "axis_t0": XA.tolist(),
                      "axis_other": YA.tolist(), "level_db": ZA.tolist()},
        "map_t0_d": {"axis_t0": GRID_T0.tolist(), "axis_d_error": GRID_D.tolist(),
                     "axis_tau_us": ((D_DESIGN + GRID_D) * TS_US).tolist(),
                     "level_db": ZB.tolist()},
        "summary": rows,
    }, indent=2, ensure_ascii=False, default=float), encoding="utf-8")

    lines = ["# 联合误差图：t_0 × T_s 与 t_0 × d", "",
             f"固定设计对 `{DESIGN_NPZ.name}`；固定对口径（录波）。"
             f"网格 21×21（Δt_0 ±2.5 μs，步长 0.25）。", "",
             "| 图 | 设计点 (dB) | 全图最小 (dB) | 最小值位置 | t_0 偏 0.5 μs | 另一维 −端 | 另一维 ＋端 | 另一维形态 |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        shape = "有内部极小" if (r["second_axis_at_minus"] > 0 and
                                r["second_axis_at_plus"] > 0) else "单调（极小在边界）"
        lines.append(f"| {r['map']} | {r['design_point_db']:.2f} | {r['min_db']:.2f} | "
                     f"(Δt_0 {r['min_at'][0]:+.2f}, 另一维 {r['min_at'][1]:+.2f}) | "
                     f"{r['t0_sensitivity_0p5']:+.2f} dB | {r['second_axis_at_minus']:+.2f} dB | "
                     f"{r['second_axis_at_plus']:+.2f} dB | {shape} |")
    lines += ["", "（\"另一维 −/＋端\"= 固定 t_0 误差为 0 时，另一维走到负端/正端相对设计点的抬升。）"]
    (OUTDIR / "结果表.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("  JSON 与结果表已保存")


if __name__ == "__main__":
    main()
