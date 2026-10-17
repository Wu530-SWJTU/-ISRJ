"""t_0 × T_s 二维误差地图（供后续曲面拟合用），z 轴 = 归一化电平 dB。

口径（与前面的图完全一致，未改动任何既有产物）：
    搜索线基线 v2、**固定设计对**（录波口径）、约束 −80 dB、零延迟原位模型
    设计对：B = 10 MHz、L = 256、t_0 = 2 μs、T_L = 1.5 μs、T_s = 6.4 μs

与既有 `结果/联合误差/` 的两点差别（本图专用）：
    1. **横纵轴用参数实际值**（t_0 ∈ [1.5, 2.5] μs、T_s ∈ [5.9, 6.9] μs），不是误差值
    2. z 轴用**归一化电平 dB**（用户口径：图上一律用 dB，不用线性幅度）

网格：11×11，步长 1 个采样点 = 0.1 μs（与三对联合误差图相同）。

    z(t_0, T_s) = 20·lg( |h^H (x ⊙ j(t_0, T_s))| / |h^H x| )   [dB]

矩阵链（零延迟原位，未变）：J(θ) = diag(j(θ))、H_j = H·diag(j(θ))、X_j 由 x⊙j 铺成。

输出（`结果/拟合/`）：
    t0_ts_电平地图.png    3D 曲面（z = 归一化电平 dB）
    t0_ts_电平地图.csv    11×11 = 121 行长表：t0_us, ts_us, level_db

用法：
    .venv/Scripts/python.exe 参数搜索/scripts/run_fit_map_t0_ts.py
"""

from __future__ import annotations

import csv
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
from isrj_joint_design.joint_design import joint_design_algorithm  # noqa: E402
from isrj_joint_design.matrices import cross_correlation  # noqa: E402
from isrj_joint_design.params import generate_isrj_sampling_function  # noqa: E402

DESIGN_NPZ = ROOT / "参数搜索" / "结果" / "参数调制" / "设计对" / "joint_design_v2_t0_2us.npz"
OUTDIR = ROOT / "参数搜索" / "结果" / "参数调制" / "拟合"
TEXT_PRIMARY, TEXT_SECONDARY = "#0b0b0b", "#52514e"
SURFACE = "#fcfcfb"

T0_AXIS = np.round(baseline.T0_US + np.arange(-0.5, 0.5001, baseline.SAMPLE_US), 4)
TS_AXIS = np.round(baseline.TS_US + np.arange(-0.5, 0.5001, baseline.SAMPLE_US), 4)


def design_or_load() -> tuple:
    if DESIGN_NPZ.exists():
        d = np.load(DESIGN_NPZ)
        print(f"  加载设计对：{DESIGN_NPZ.name}")
        return d["x"], d["h"]
    print("  设计对不存在，跑一次联合设计 ...")
    DESIGN_NPZ.parent.mkdir(parents=True, exist_ok=True)
    res = joint_design_algorithm(baseline.params_for(), verbose=False)
    np.savez_compressed(DESIGN_NPZ, x=res["x"], h=res["h"], j=res["j"])
    return res["x"], res["h"]


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    OUTDIR.mkdir(parents=True, exist_ok=True)
    print("=== t_0 × T_s 二维误差地图（实际参数值，z = 归一化电平 dB）===")
    print("  " + baseline.summary().replace(chr(10), chr(10) + "  "))

    x, h = design_or_load()
    peak = abs(np.vdot(h, x))

    def level_db(t0_us: float, ts_us: float) -> float:
        p = baseline.params_for(t0_us=t0_us, ts_us=ts_us)
        r = cross_correlation(x * generate_isrj_sampling_function(p), h)
        return float(20 * np.log10(abs(r[baseline.N - 1]) / peak + 1e-30))

    Z = np.array([[level_db(a, b) for b in TS_AXIS] for a in T0_AXIS])

    i0 = int(np.argmin(np.abs(T0_AXIS - baseline.T0_US)))
    j0 = int(np.argmin(np.abs(TS_AXIS - baseline.TS_US)))
    gi = np.unravel_index(Z.argmax(), Z.shape)
    print(f"  设计点（t_0 = {baseline.T0_US:g}、T_s = {baseline.TS_US:g}）电平 = "
          f"{Z[i0, j0]:.2f} dB")
    print(f"  全图最高 = {Z.max():.2f} dB @ t_0 = {T0_AXIS[gi[0]]:.2f}、"
          f"T_s = {TS_AXIS[gi[1]]:.2f}；全图最低 = {Z.min():.2f} dB")

    # ---- 数据（长表，直接给拟合用） ----
    csv_path = OUTDIR / "t0_ts_电平地图.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["t0_us", "ts_us", "level_db"])
        for a, row in zip(T0_AXIS, Z):
            for b, v in zip(TS_AXIS, row):
                w.writerow([f"{a:.2f}", f"{b:.2f}", f"{v:.4f}"])
    print(f"  数据已保存：{csv_path.name}（{Z.size} 行）")

    # ---- 3D 曲面 ----
    T0g, TSg = np.meshgrid(T0_AXIS, TS_AXIS, indexing="ij")
    fig = plt.figure(figsize=(9.5, 7), facecolor=SURFACE)
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(T0g, TSg, Z, cmap="viridis", edgecolor="#ffffff",
                           linewidth=0.4, antialiased=True, rstride=1, cstride=1)
    ax.set_xlabel("$t_0$ (μs)", color=TEXT_SECONDARY, labelpad=8)
    ax.set_ylabel("$T_s$ (μs)", color=TEXT_SECONDARY, labelpad=8)
    ax.set_zlabel("Normalized level (dB)", color=TEXT_SECONDARY, labelpad=8)
    ax.set_title(f"Level after filtering (design constraint {baseline.PEAK_INTERF_DB:g} dB)",
                 color=TEXT_PRIMARY, fontsize=12)
    ax.view_init(elev=26, azim=-56)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_facecolor(SURFACE)
        pane.pane.set_edgecolor("#d8d7d2")
    fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.08, label="Level (dB)")
    out = OUTDIR / "t0_ts_电平地图.png"
    fig.savefig(out, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"  图已保存：{out.name}")


if __name__ == "__main__":
    main()
