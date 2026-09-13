"""占空比误差一维扫描：归一化电平 – Δd（设计点 d = 0.25）。

搜索线参数化 θ = [t_0, d, T_s]，`T_L = d·T_s` 为派生量；固定 `T_s` = 10 μs、
`t_0` = 2.5 μs，扫占空比 d（等价于扫截取长度 T_L，ΔT_L = Δd·T_s）。

口径与其它一维实验一致（固定设计对，录波口径）：加载设计对
（t_0 = 2.5 μs、T_s = 10 μs、T_L = 2.5 μs），扫**实际**干扰的占空比：

    实际干扰  y_j(θ*) = x ⊙ j(θ*)，  θ* = (t_0, T_L = (0.25+Δd)·T_s, T_s)
    电平      z = 20lg( |h^H(x ⊙ j(θ*))| / |h^H x| )

矩阵链（未变）：J(θ) = diag(j(θ))、H_j = H·diag(j(θ))、X_j 由 x⊙j 铺成；
本参数下仅 `N_tau` 随 Δd 变化（窗口宽度），窗口起点与周期不变。

扫描范围：Δd ∈ [−0.20, +0.20]（d ∈ [0.05, 0.45]，即 T_L ∈ [0.5, 4.5] μs），
步长 0.01（41 点）。设计点 d = 0.25 处约束钉死 −30 dB。

用法：
    .venv/Scripts/python.exe 参数搜索/scripts/run_duty_sweep.py
"""

from __future__ import annotations

import argparse
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
T0_US, TS_US = 2.5, 10.0
D_DESIGN = 0.25
DESIGN_NPZ = ROOT / "参数搜索" / "结果" / "窗口起点" / "joint_design_result_t0_2.5us.npz"
OUTDIR = ROOT / "参数搜索" / "结果" / "占空比"
SERIES_1 = "#2a78d6"
SERIES_2 = "#eb6834"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
SURFACE = "#fcfcfb"
GRID = "#d8d7d2"


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


def sweep(x: np.ndarray, h: np.ndarray, dd: np.ndarray) -> list:
    peak = abs(np.vdot(h, x))
    rows = []
    for e in dd:
        d = D_DESIGN + float(e)
        p = GlobalParams(N=N, tau=d * TS_US * 1e-6, T_s=TS_US * 1e-6,
                         tau_offset=T0_US * 1e-6, seed=0)
        j_actual = generate_isrj_sampling_function(p)
        r = cross_correlation(x * j_actual, h)
        zero_db = float(20 * np.log10(abs(r[N - 1]) / peak + 1e-30))
        peak_db = float(20 * np.log10(np.abs(r).max() / peak + 1e-30))
        # 参考口径二：相对"干扰自身的匹配滤波峰"= Σ j（|x|=1，故匹配峰 = 采样点数）
        own_matched = float(np.sum(j_actual))
        rel_own_db = float(20 * np.log10(abs(r[N - 1]) / max(own_matched, 1e-30) + 1e-30))
        rows.append({
            "d": round(d, 6),
            "d_error": round(float(e), 6),
            "tau_us": round(d * TS_US, 6),
            "zero_delay_db": zero_db,
            "peak_db": peak_db,
            "rel_own_matched_db": rel_own_db,
            "jamming_samples": own_matched,
            "C": float(10 ** (peak_db / 20)),
            "il": float(np.sum(np.abs(r) ** 2)),
        })
    return rows


def plot(rows: list, save_path: Path) -> None:
    dd = np.array([r["d_error"] for r in rows])
    zero = np.array([r["zero_delay_db"] for r in rows])
    peak = np.array([r["peak_db"] for r in rows])
    il = np.array([r["il"] for r in rows])
    rel_own = np.array([r["rel_own_matched_db"] for r in rows])

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), facecolor=SURFACE)

    ax = axes[0]
    ax.plot(dd, zero, color=SERIES_1, linewidth=2, marker="o", markersize=4,
            label="zero-delay level (rel. mainlobe)")
    ax.plot(dd, peak, color=SERIES_2, linestyle="none", marker="s", markersize=3,
            label="overall peak (= zero-delay)")
    ax.plot(dd, rel_own, color=SERIES_2, linewidth=2, linestyle="--",
            label="suppression rel. jammer's own matched peak")
    ax.axhline(-30, color=TEXT_SECONDARY, linewidth=1.0, linestyle="-.")
    ax.axvline(0.0, color=TEXT_SECONDARY, linewidth=1.0, linestyle="--", alpha=0.8)
    ax.annotate(f"design point d = {D_DESIGN:g} (T_L = {D_DESIGN * TS_US:g} μs)\n"
                f"zero-delay at design = {zero[np.argmin(np.abs(dd))]:.2f} dB\n"
                f"T_L = d × T_s,  T_s = {TS_US:g} μs",
                xy=(0.03, 0.06), xycoords="axes fraction",
                fontsize=9, color=TEXT_PRIMARY)
    ax.set_xlabel("Duty-cycle error Δd")
    ax.set_ylabel("Normalized level (dB)")
    ax.set_title("(a) ISRJ level after filtering, under duty-cycle error", fontsize=11)
    ax.legend(frameon=False, fontsize=9, labelcolor=TEXT_SECONDARY)
    _style(ax)

    ax = axes[1]
    ax.semilogy(dd, il, color=SERIES_1, linewidth=2, marker="o", markersize=4)
    ax.axvline(0.0, color=TEXT_SECONDARY, linewidth=1.0, linestyle="--", alpha=0.8)
    ax.set_xlabel("Duty-cycle error Δd")
    ax.set_ylabel("IL")
    ax.set_title("(b) Integrated jamming level under duty-cycle error", fontsize=11)
    _style(ax)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"  图已保存：{save_path.name}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--min", type=float, default=-0.20, help="Δd 下界")
    ap.add_argument("--max", type=float, default=0.20, help="Δd 上界")
    ap.add_argument("--step", type=float, default=0.01, help="Δd 步长")
    args = ap.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    OUTDIR.mkdir(parents=True, exist_ok=True)

    d = np.load(DESIGN_NPZ)
    x, h = d["x"], d["h"]
    print("=== 占空比误差一维扫描（搜索线）===")
    print(f"  设计对：{DESIGN_NPZ.name}（t_0 = {T0_US:g} μs、T_s = {TS_US:g} μs、"
          f"d = {D_DESIGN:g}）")
    print(f"  扫描 Δd ∈ [{args.min:g}, {args.max:g}]，步长 {args.step:g}"
          f"（T_L ∈ [{(D_DESIGN+args.min)*TS_US:g}, {(D_DESIGN+args.max)*TS_US:g}] μs）")

    dd = np.arange(args.min, args.max + args.step / 2, args.step)
    rows = sweep(x, h, dd)
    i0 = int(np.argmin(np.abs(np.array([r["d_error"] for r in rows]))))
    print(f"  设计点 Δd = 0（d = {rows[i0]['d']:.2f}，T_L = {rows[i0]['tau_us']:.2f} μs）："
          f"{rows[i0]['zero_delay_db']:.2f} dB、C = {rows[i0]['C']:.4f}")
    for e in (args.min, args.min / 2, args.max / 2, args.max):
        k = int(np.argmin(np.abs(np.array([r["d_error"] for r in rows]) - e)))
        print(f"  Δd = {rows[k]['d_error']:+.2f}（d = {rows[k]['d']:.2f}，"
              f"T_L = {rows[k]['tau_us']:.2f} μs）：{rows[k]['zero_delay_db']:.2f} dB"
              f"（抬升 {rows[k]['zero_delay_db'] - rows[i0]['zero_delay_db']:.2f} dB）")

    plot(rows, OUTDIR / "归一化电平_占空比.png")
    print(f"  参考口径二（相对干扰自身匹配峰）在设计点 "
          f"{rows[i0]['rel_own_matched_db']:.2f} dB；"
          f"Δd = {args.min:+.2f} 处 {rows[int(np.argmin(np.abs(dd - args.min)))]['rel_own_matched_db']:.2f} dB")
    (OUTDIR / "duty_sweep.json").write_text(json.dumps({
        "design": {"npz": DESIGN_NPZ.name, "t0_us": T0_US, "ts_us": TS_US,
                   "d": D_DESIGN, "N": N, "seed": 0, "init": "random_phase"},
        "d_error": [r["d_error"] for r in rows],
        "d": [r["d"] for r in rows],
        "tau_us": [r["tau_us"] for r in rows],
        "zero_delay_db": [r["zero_delay_db"] for r in rows],
        "peak_db": [r["peak_db"] for r in rows],
        "rel_own_matched_db": [r["rel_own_matched_db"] for r in rows],
        "jamming_samples": [r["jamming_samples"] for r in rows],
        "C": [r["C"] for r in rows],
        "il": [r["il"] for r in rows],
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = ["# 占空比误差一维扫描（设计点 d = 0.25、T_s = 10 μs、t_0 = 2.5 μs）", "",
             f"扫描 Δd ∈ [{args.min:g}, {args.max:g}]，步长 {args.step:g}；"
             f"T_L = d·T_s。归一化分母 = |h^H x| = b_max = N。", "",
             "| Δd | d | T_L (μs) | 零延迟电平 (dB) | C | IL |", "|---|---|---|---|---|---|"]
    for r in rows:
        if abs(r["d_error"] * 20 - round(r["d_error"] * 20)) > 1e-6:
            continue                       # 结果表每 0.05 取一行
        lines.append(f"| {r['d_error']:+.2f} | {r['d']:.2f} | {r['tau_us']:.2f} | "
                     f"{r['zero_delay_db']:.2f} | {r['C']:.3f} | {r['il']:.3e} |")
    (OUTDIR / "结果表.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  JSON 与结果表已保存")


if __name__ == "__main__":
    main()
