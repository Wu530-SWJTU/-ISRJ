"""二维参数误差可视化：转发时间误差 × 采样周期误差 → 归一化干扰电平。

导师要求的多维可视化。x = 转发时间（= 转发起始时间 `t_0`、采样窗口起点，
复现线里叫 `tau_offset`，即"转发时间估计误差"那一族）误差，y = 采样周期
`T_s` 误差，z = 归一化干扰电平（dB，相对主瓣；按 J/S = 0 dB 幅度口径可直接
读作"假目标相对真目标"）。

口径与 1D 实验完全一致：**固定设计对**（录波口径）——用 t_0 = 2.5 μs、
T_s = 10 μs 的设计对，扫实际干扰参数，误差只进实际干扰：

    实际干扰  y_j(θ*) = x ⊙ j(θ*)，  θ* = (t_0 + ε_t0, T_L, T_s + ε_Ts)
    电平      z = 20lg( |h^H(x ⊙ j(θ*))| / |h^H x| )

矩阵链（未变）：J(θ) = diag(j(θ))、H_j = H·diag(j(θ))、X_j 由 x⊙j 铺成。

网格：ε_t0、ε_Ts ∈ [−2.5, +2.5] μs，步长 0.25（各 21 点，共 441 点）。
合法性：T_L = 2.5 μs 固定 ⇒ 需 T_s ≥ t_0 + T_L；本网格内 T_s ∈ [7.5, 12.5]、
t_0 ∈ [0, 5]，全部合法（窗口不跨周期）。

用法：
    .venv/Scripts/python.exe 参数搜索/scripts/run_2d_error_map.py
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
GRID = np.arange(-2.5, 2.5001, 0.25)
DESIGN_NPZ = ROOT / "参数搜索" / "结果" / "窗口起点" / "joint_design_result_t0_2.5us.npz"
OUTDIR = ROOT / "参数搜索" / "结果" / "联合误差"
SURFACE_C = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID_C = "#d8d7d2"


def level_db(x: np.ndarray, h: np.ndarray, e_t0: float, e_ts: float) -> float:
    p = GlobalParams(N=N, tau=TAU_US * 1e-6, T_s=(TS_US + e_ts) * 1e-6,
                     tau_offset=(T0_US + e_t0) * 1e-6, seed=0)
    j = generate_isrj_sampling_function(p)
    r = cross_correlation(x * j, h)
    return float(20 * np.log10(abs(r[N - 1]) / abs(np.vdot(h, x)) + 1e-30))


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    OUTDIR.mkdir(parents=True, exist_ok=True)

    d = np.load(DESIGN_NPZ)
    x, h = d["x"], d["h"]
    peak = abs(np.vdot(h, x))
    print("=== 二维误差图：转发时间误差 × 采样周期误差 → 归一化电平 ===")
    print(f"  设计对：{DESIGN_NPZ.name}（t_0 = {T0_US:g} μs、T_s = {TS_US:g} μs、"
          f"T_L = {TAU_US:g} μs），归一化分母 |h^H x| = {peak:.2f}")

    Z = np.array([[level_db(x, h, e0, ets) for ets in GRID] for e0 in GRID])
    i0 = int(np.argmin(np.abs(GRID)))
    print(f"  网格 {GRID.size}×{GRID.size} = {GRID.size ** 2} 点"
          f"（ε ∈ ±2.5 μs，步长 0.25）")
    print(f"  设计点 (0, 0)：{Z[i0, i0]:.2f} dB")
    print(f"  谷底位置：ε_t0 = {GRID[np.unravel_index(Z.argmin(), Z.shape)[0]]:+.2f} μs、"
          f"ε_Ts = {GRID[np.unravel_index(Z.argmin(), Z.shape)[1]]:+.2f} μs"
          f"（{Z.min():.2f} dB）")
    j05 = int(np.argmin(np.abs(GRID - 0.5)))
    print(f"  灵敏度对比（各偏 0.5 μs）：转发时间误差 → {Z[j05, i0]:.2f} dB ；"
          f"采样周期误差 → {Z[i0, j05]:.2f} dB")
    print(f"  四角：{Z[0, 0]:.2f} / {Z[0, -1]:.2f} / {Z[-1, 0]:.2f} / {Z[-1, -1]:.2f} dB")

    # 谷脊：每个 ε_Ts 下沿 ε_t0 的最优位置；补偿量 = 该列最优 − 该列 t_0 误差为 0 的值
    ridge = Z.argmin(axis=0)
    ridge_t0 = GRID[ridge]
    slope, intercept = np.polyfit(GRID, ridge_t0, 1)
    resid = float(np.max(np.abs(ridge_t0 - (slope * GRID + intercept))))
    best_over_t0 = Z.min(axis=0)
    comp = Z[i0, :] - best_over_t0          # 固定 T_s 误差时，调 t_0 最多能挽回多少 dB
    print(f"  谷脊（每个 T_s 误差下 t_0 误差的最优值）：拟合斜率 {slope:+.3f}"
          f"（最大偏离直线 {resid:.2f} μs，谷底在 |ε_Ts| ≳ 2 μs 处已出网格）")
    print(f"  两维耦合强度（固定 T_s 误差、调 t_0 最多挽回的 dB）：")
    for ets in (-2.0, -1.0, -0.5, 0.5, 1.0, 2.0):
        k = int(np.argmin(np.abs(GRID - ets)))
        print(f"    ε_Ts = {ets:+.1f} μs：t_0 误差为 0 时 {Z[i0, k]:.2f} dB → "
              f"最优 t_0 误差 {ridge_t0[k]:+.2f} μs 时 {best_over_t0[k]:.2f} dB"
              f"（挽回 {comp[k]:.2f} dB）")
    k_best = int(np.argmax(comp))
    comp_max = float(comp[k_best])
    deg = float(Z[i0, k_best] + 30.0)        # 该点相对 -30 dB 基准的退化
    frac = comp_max / deg
    print(f"  最大补偿量 = {comp_max:.2f} dB（在 ε_Ts = {GRID[k_best]:+.2f} μs 处，"
          f"该点总退化 {deg:.2f} dB，占 {frac * 100:.0f}%）"
          f" ⇒ 部分耦合：另一维能挽回约三分之一")

    X, Y = np.meshgrid(GRID, GRID, indexing="ij")

    fig = plt.figure(figsize=(15.5, 6.0), facecolor=SURFACE_C)

    ax = fig.add_subplot(1, 2, 1, projection="3d")
    surf = ax.plot_surface(X, Y, Z, cmap="viridis", linewidth=0, antialiased=True,
                           rstride=1, cstride=1, alpha=0.95)
    ax.contour(X, Y, Z, levels=12, zdir="z", offset=Z.min() - 3,
               cmap="viridis", linewidths=0.7)
    ax.set_xlabel("Forwarding-start error (μs)", labelpad=8, fontsize=9)
    ax.set_ylabel("Sampling-period error (μs)", labelpad=8, fontsize=9)
    ax.set_zlabel("Normalized level (dB)", labelpad=6, fontsize=9)
    ax.set_title("(a) Normalized ISRJ level after filtering", fontsize=11,
                 color=TEXT_PRIMARY)
    ax.tick_params(labelsize=8, colors=TEXT_SECONDARY)
    ax.view_init(elev=26, azim=-56)
    fig.colorbar(surf, ax=ax, shrink=0.62, pad=0.08, label="level (dB)")

    ax2 = fig.add_subplot(1, 2, 2)
    cf = ax2.contourf(X, Y, Z, levels=18, cmap="viridis")
    cs = ax2.contour(X, Y, Z, levels=[-25, -20, -15, -12, -10],
                     colors="white", linewidths=0.9)
    ax2.clabel(cs, fmt="%d dB", fontsize=8, colors="white")
    ax2.plot(GRID, ridge_t0, color="#eb6834", linewidth=0, marker="o",
             markersize=4, markeredgecolor="white", markeredgewidth=0.6,
             label=f"argmin over t_0 (ridge slope {slope:+.2f})")
    ax2.plot(0, 0, marker="x", color="#eb6834", markersize=9, markeredgewidth=2)
    ax2.legend(loc="lower right", frameon=False, fontsize=9, labelcolor=TEXT_PRIMARY)
    ax2.annotate(f"design point\n{Z[i0, i0]:.1f} dB", xy=(0, 0), xytext=(8, 10),
                 textcoords="offset points", fontsize=9, color=TEXT_PRIMARY)
    ax2.set_xlabel("Forwarding-start error (μs)")
    ax2.set_ylabel("Sampling-period error (μs)")
    ax2.set_title("(b) Same grid as contours", fontsize=11, color=TEXT_PRIMARY)
    ax2.tick_params(labelsize=9, colors=TEXT_SECONDARY)
    ax2.set_facecolor(SURFACE_C)
    for side in ("top", "right"):
        ax2.spines[side].set_visible(False)
    ax2.grid(True, color=GRID_C, linewidth=0.6, alpha=0.6)
    ax2.set_axisbelow(True)
    fig.colorbar(cf, ax=ax2, shrink=0.9, pad=0.03, label="level (dB)")

    fig.tight_layout()
    out = OUTDIR / "t0_ts_联合误差.png"
    fig.savefig(out, dpi=150, facecolor=SURFACE_C)
    plt.close(fig)
    print(f"  图已保存：{out.name}")

    (OUTDIR / "t0_ts_联合误差.json").write_text(json.dumps({
        "design": {"npz": DESIGN_NPZ.name, "t0_us": T0_US, "ts_us": TS_US,
                   "tau_us": TAU_US, "N": N, "seed": 0, "init": "random_phase",
                   "denominator": float(peak)},
        "axis": {"forwarding_start_error_us": GRID.tolist(),
                 "sampling_period_error_us": GRID.tolist()},
        "level_db": Z.tolist(),
        "dims": {
            "forwarding_start": "t_0 / tau_offset（转发起始时间 = 采样窗口起点）",
            "sampling_period": "T_s（采样重复周期）",
        },
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = ["# 二维误差图：转发时间误差 × 采样周期误差 → 归一化电平", "",
             f"固定设计对 `{DESIGN_NPZ.name}`（t_0 = {T0_US:g} μs、T_s = {TS_US:g} μs、"
             f"T_L = {TAU_US:g} μs）；网格 ±2.5 μs、步长 0.25（21×21）。", "",
             "| 指标 | 值 |", "|---|---|",
             f"| 设计点 (0, 0) | {Z[i0, i0]:.2f} dB |",
             f"| 谷底位置 | (ε_t0, ε_Ts) = ({GRID[np.unravel_index(Z.argmin(), Z.shape)[0]]:+.2f}, "
             f"{GRID[np.unravel_index(Z.argmin(), Z.shape)[1]]:+.2f}) μs，{Z.min():.2f} dB |",
             f"| 偏 0.5 μs：转发时间维 | {Z[j05, i0]:.2f} dB |",
             f"| 偏 0.5 μs：采样周期维 | {Z[i0, j05]:.2f} dB |",
             f"| 四角 (∓2.5, ∓2.5) | {Z[0, 0]:.2f} / {Z[0, -1]:.2f} / {Z[-1, 0]:.2f} / {Z[-1, -1]:.2f} dB |",
             f"| 谷脊斜率（t_0 误差对 T_s 误差的最优响应） | {slope:+.3f}（最大偏离直线 {resid:.2f} μs） |",
             f"| 两维耦合强度（固定 T_s 误差、调 t_0 的最大补偿量） | {comp_max:.2f} dB（约占退化的 {frac * 100:.0f}%） |",
             "", "按 J/S = 0 dB 幅度口径，表中的 dB 可直接读作「假目标相对真目标」。"]
    (OUTDIR / "t0_ts_结果表.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  JSON 与结果表已保存")


if __name__ == "__main__":
    main()
