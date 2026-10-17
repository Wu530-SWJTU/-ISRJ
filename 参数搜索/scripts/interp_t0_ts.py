"""t_0 × T_s 结果表的二维插值（供搜索在网格内部取值用）。

数据：`结果/拟合/t0_ts_电平地图.csv`（121 个真实算出来的格点，11×11，步长 0.1 μs）

方法：**双线性插值**（`scipy.interpolate.RegularGridInterpolator`, method="linear"）
    - 在 121 个格点上严格复现原值（自检见下）
    - 格内按线性过渡（格子里原本是量化平台，插值相当于给定"台阶之间的过渡"）
    - 不用三次样条：本数据动态范围 68 dB 且谷是尖 V，三次会过冲、在谷底造出假振荡

接口（搜索直接调用）：

    from interp_t0_ts import load_interpolator
    f = load_interpolator()        # f(t0_us, ts_us) -> 电平 dB
    f(2.2, 6.55)

命令行（生成插值曲面图 + 自检 + 格内误差抽查）：
    .venv/Scripts/python.exe 参数搜索/scripts/interp_t0_ts.py
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
from scipy.interpolate import RegularGridInterpolator  # noqa: E402

import baseline  # noqa: E402

CSV_PATH = ROOT / "参数搜索" / "结果" / "参数调制" / "拟合" / "t0_ts_电平地图.csv"
OUTDIR = ROOT / "参数搜索" / "结果" / "参数调制" / "拟合"
TEXT_PRIMARY, TEXT_SECONDARY = "#0b0b0b", "#52514e"
SURFACE = "#fcfcfb"

# 格内误差抽查用的点（不在网格上）
PROBE = [(2.20, 6.55), (2.05, 6.32), (2.23, 6.42)]


def load_table(path: Path = CSV_PATH) -> tuple:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    t0 = np.array(sorted({float(r["t0_us"]) for r in rows}))
    ts = np.array(sorted({float(r["ts_us"]) for r in rows}))
    Z = np.full((t0.size, ts.size), np.nan)
    for r in rows:
        i, j = int(np.argmin(np.abs(t0 - float(r["t0_us"])))), \
               int(np.argmin(np.abs(ts - float(r["ts_us"]))))
        Z[i, j] = float(r["level_db"])
    assert not np.isnan(Z).any(), "结果表有缺口"
    return t0, ts, Z


def load_interpolator(path: Path = CSV_PATH):
    """返回 f(t0_us, ts_us) -> 电平 dB。范围外报错（不外推）。"""
    t0, ts, Z = load_table(path)
    it = RegularGridInterpolator((t0, ts), Z, method="linear", bounds_error=True)

    def f(t0_us: float, ts_us: float) -> float:
        return float(it([[t0_us, ts_us]])[0])

    return f


def true_level(t0_us: float, ts_us: float) -> float:
    """真实算一次（对照用）：固定设计对上跑一次匹配滤波。"""
    from isrj_joint_design.matrices import cross_correlation
    from isrj_joint_design.params import generate_isrj_sampling_function

    d = np.load(ROOT / "参数搜索" / "结果" / "参数调制" / "设计对" / "joint_design_v2_t0_2us.npz")
    x, h = d["x"], d["h"]
    peak = abs(np.vdot(h, x))
    r = cross_correlation(x * generate_isrj_sampling_function(
        baseline.params_for(t0_us=t0_us, ts_us=ts_us)), h)
    return float(20 * np.log10(abs(r[baseline.N - 1]) / peak + 1e-30))


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    t0, ts, Z = load_table()
    it = RegularGridInterpolator((t0, ts), Z, method="linear", bounds_error=True)
    print("=== t_0 × T_s 二维插值（双线性）===")
    print(f"  数据：{CSV_PATH.name}，{Z.size} 个格点（{t0.size}×{ts.size}）")

    T0g, TSg = np.meshgrid(t0, ts, indexing="ij")
    back = it(np.column_stack([T0g.ravel(), TSg.ravel()])).reshape(Z.shape)
    print(f"  自检（121 个格点）：插值与原值最大偏差 = {np.abs(back - Z).max():.2e} dB")

    print("  格内抽查（插值 vs 真算）：")
    for a, b in PROBE:
        v_i, v_t = float(it([[a, b]])[0]), true_level(a, b)
        print(f"    t_0={a:.2f}, T_s={b:.2f}：插值 {v_i:7.2f} dB ；真算 {v_t:7.2f} dB"
              f" ；差 {v_i - v_t:+.2f} dB")

    # ---- 插值曲面（0.01 μs 细网格） ----
    fine0 = np.round(np.arange(t0[0], t0[-1] + 1e-9, 0.01), 4)
    fineT = np.round(np.arange(ts[0], ts[-1] + 1e-9, 0.01), 4)
    F0, FT = np.meshgrid(fine0, fineT, indexing="ij")
    ZF = it(np.column_stack([F0.ravel(), FT.ravel()])).reshape(F0.shape)

    fig = plt.figure(figsize=(9.5, 7), facecolor=SURFACE)
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(F0, FT, ZF, cmap="viridis", linewidth=0,
                           antialiased=True, rcount=101, ccount=101)
    ax.set_xlabel("$t_0$ (μs)", color=TEXT_SECONDARY, labelpad=8)
    ax.set_ylabel("$T_s$ (μs)", color=TEXT_SECONDARY, labelpad=8)
    ax.set_zlabel("Normalized level (dB)", color=TEXT_SECONDARY, labelpad=8)
    ax.set_title("Bilinear interpolation of the t_0 × T_s level table",
                 color=TEXT_PRIMARY, fontsize=12)
    ax.view_init(elev=26, azim=-56)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_facecolor(SURFACE)
        pane.pane.set_edgecolor("#d8d7d2")
    fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.08, label="Level (dB)")
    out = OUTDIR / "t0_ts_电平_插值曲面.png"
    fig.savefig(out, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"  图已保存：{out.name}")


if __name__ == "__main__":
    main()
