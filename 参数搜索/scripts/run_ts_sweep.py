"""采样周期误差一维扫描（单配置）：归一化电平 – ΔT_s。

口径：搜索线基线 v2（`scripts/baseline.py`）的**信号侧与干扰侧参数**（B = 10 MHz、
L = 256、t_0 = 2 μs、T_L = 1.5 μs、T_s = 6.4 μs），约束深度可用 `--depth` 指定
（默认跟随基线 `baseline.PEAK_INTERF_DB` = **−80 dB**）；初始化 random_phase、seed 0。

固定设计对（录波口径）：用该约束下的设计对评估**实际**采样周期偏离时的性能：

    y_j(θ*) = x ⊙ j(θ*)，  z = 20lg( |h^H y_j(θ*)| / |h^H x| )

矩阵链（零延迟原位，未变）：J(θ) = diag(j(θ))、H_j = H·diag(j(θ))、X_j 由 x⊙j 铺成。

误差范围：**±p% × T_s**（`--err-percent`，默认 10% ⇒ ±0.64 μs），步长 `--step-us`（默认 0.04 μs）。

用法：
    .venv/Scripts/python.exe 参数搜索/scripts/run_ts_sweep.py
    .venv/Scripts/python.exe 参数搜索/scripts/run_ts_sweep.py --depth -30 --err-percent 10  # 换约束深度
"""

from __future__ import annotations

import argparse
import json
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
from isrj_joint_design.joint_design import compute_metrics, joint_design_algorithm  # noqa: E402
from isrj_joint_design.matrices import cross_correlation  # noqa: E402
from isrj_joint_design.params import generate_isrj_sampling_function  # noqa: E402

DESIGN_DIR = ROOT / "参数搜索" / "结果" / "设计对"
OUTDIR = ROOT / "参数搜索" / "结果" / "采样周期扫描"
SERIES_1, SERIES_2 = "#2a78d6", "#eb6834"
TEXT_PRIMARY, TEXT_SECONDARY = "#0b0b0b", "#52514e"
SURFACE, GRID_C = "#fcfcfb", "#d8d7d2"


def design(depth: float) -> tuple:
    npz = DESIGN_DIR / f"joint_design_v2_t0_2us_bmin{abs(depth):g}dB.npz"
    if npz.exists():
        d = np.load(npz)
        print(f"  加载设计对：{npz.name}")
        return d["x"], d["h"], d["j"], None
    print(f"  联合设计（约束 {depth:g} dB）...")
    DESIGN_DIR.mkdir(parents=True, exist_ok=True)
    res = joint_design_algorithm(baseline.params_for(peak_interf_db=depth), verbose=False)
    np.savez_compressed(npz, x=res["x"], h=res["h"], j=res["j"])
    print(f"    {res['elapsed_s']:.1f} s")
    return res["x"], res["h"], res["j"], res


def style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID_C, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    ax.xaxis.label.set_color(TEXT_SECONDARY)
    ax.yaxis.label.set_color(TEXT_SECONDARY)
    ax.title.set_color(TEXT_PRIMARY)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--depth", type=float, default=baseline.PEAK_INTERF_DB,
                    help="零延迟干扰约束 [dB]（默认跟随基线）")
    ap.add_argument("--err-percent", type=float, default=10.0,
                    help="误差范围 = ±p%% × T_s（默认 10%%）")
    ap.add_argument("--step-us", type=float, default=0.04, help="误差步长 [μs]")
    args = ap.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    OUTDIR.mkdir(parents=True, exist_ok=True)

    span = baseline.TS_US * args.err_percent / 100.0
    err = np.arange(-span, span + args.step_us / 2, args.step_us)
    print(f"=== 采样周期误差扫描（约束 {args.depth:g} dB，误差 ±{span:g} μs "
          f"= ±{args.err_percent:g}% × T_s）===")
    print("  " + baseline.summary().replace(chr(10), chr(10) + "  "))

    x, h, j, res = design(args.depth)
    m = compute_metrics(x, h, j)
    peak = abs(np.vdot(h, x))
    print(f"  设计对指标：PSL {m['psl_db']:.2f} dB、损失 {m['processing_gain_loss_db']:.2f} dB、"
          f"零延迟 {m['isrj_zero_delay_db']:.2f} dB")

    rows = []
    for e in err:
        p = baseline.params_for(ts_us=baseline.TS_US + float(e), peak_interf_db=args.depth)
        r = cross_correlation(x * generate_isrj_sampling_function(p), h)
        zero = float(20 * np.log10(abs(r[baseline.N - 1]) / peak + 1e-30))
        rows.append({"ts_error_us": round(float(e), 6),
                     "ts_us": round(baseline.TS_US + float(e), 6),
                     "zero_delay_db": zero,
                     "peak_db": float(20 * np.log10(np.abs(r).max() / peak + 1e-30)),
                     "rise_db": float(zero - m["isrj_zero_delay_db"]),
                     "amp_ratio": float(10 ** (zero / 20)),   # 线性幅度比 = |h^H y_j| / |h^H x|
                     "il": float(np.sum(np.abs(r) ** 2))})
    i0 = int(np.argmin(np.abs(err)))
    print(f"  谷底（ε=0）：{rows[i0]['zero_delay_db']:.2f} dB；"
          f"两端 ε=±{span:g} μs：{rows[0]['zero_delay_db']:.2f} / {rows[-1]['zero_delay_db']:.2f} dB")
    k10 = int(np.argmin(np.abs(err - 0.1 * np.sign(1))))
    print(f"  ±0.1 μs（1 采样点）处：{rows[k10]['zero_delay_db']:.2f} dB"
          f"（抬升 {rows[k10]['rise_db']:.2f} dB）")

    fig, axes = plt.subplots(1, 3, figsize=(18.5, 4.7), facecolor=SURFACE)
    e = np.array([r["ts_error_us"] for r in rows])
    ax = axes[0]
    ax.plot(e, [r["zero_delay_db"] for r in rows], color=SERIES_1, linewidth=2,
            marker="o", markersize=4, label="zero-delay level")
    ax.plot(e, [r["peak_db"] for r in rows], color=SERIES_2, linestyle="none",
            marker="s", markersize=3, label="overall peak (= zero-delay)")
    ax.axhline(args.depth, color=TEXT_SECONDARY, linewidth=1.0, linestyle="-.")
    ax.axvline(0.0, color=TEXT_SECONDARY, linewidth=1.0, linestyle="--", alpha=0.8)
    ax.set_xlabel("Sampling-period error (μs)")
    ax.set_ylabel("Normalized level (dB)")
    ax.set_title(f"(a) Level after filtering (design constraint {args.depth:g} dB)",
                 fontsize=11)
    ax.annotate(f"error range = ±{args.err_percent:g}% × $T_s$ = ±{span:g} μs;  "
                f"valley = {rows[i0]['zero_delay_db']:.2f} dB",
                xy=(0.03, 0.06), xycoords="axes fraction", fontsize=9,
                color=TEXT_PRIMARY)
    ax.legend(frameon=False, fontsize=9, labelcolor=TEXT_SECONDARY)
    style(ax)

    ax = axes[1]
    ax.plot(e, [r["amp_ratio"] for r in rows], color=SERIES_1, linewidth=2,
            marker="o", markersize=4)
    ax.axvline(0.0, color=TEXT_SECONDARY, linewidth=1.0, linestyle="--", alpha=0.8)
    ax.set_xlabel("Sampling-period error (μs)")
    ax.set_ylabel("Amplitude ratio  $|h^H y_j| / |h^H x|$")
    ax.set_title("(b) Linear amplitude of the false target (rel. true target)",
                 fontsize=11)
    ax.annotate(f"zero error: {rows[i0]['amp_ratio']:.2e};  "
                f"±{span:g} μs: {rows[0]['amp_ratio']:.2e} / {rows[-1]['amp_ratio']:.2e}",
                xy=(0.03, 0.75), xycoords="axes fraction", fontsize=9,
                color=TEXT_PRIMARY)
    style(ax)

    ax = axes[2]
    ax.semilogy(e, [r["il"] for r in rows], color=SERIES_1, linewidth=2, marker="o",
                markersize=4)
    ax.axvline(0.0, color=TEXT_SECONDARY, linewidth=1.0, linestyle="--", alpha=0.8)
    ax.set_xlabel("Sampling-period error (μs)")
    ax.set_ylabel("IL")
    ax.set_title("(c) Integrated jamming level", fontsize=11)
    style(ax)

    fig.tight_layout()
    out = OUTDIR / f"采样周期误差扫描_约束{abs(args.depth):g}dB.png"
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"  图已保存：{out.name}")

    (OUTDIR / f"ts_sweep_{abs(args.depth):g}dB.json").write_text(json.dumps({
        "config": {"depth_db": args.depth, "err_percent": args.err_percent,
                   "span_us": span, "step_us": args.step_us,
                   "B_hz": baseline.B_HZ, "N": baseline.N, "T_p_s": baseline.T_P_S,
                   "t0_us": baseline.T0_US, "tau_us": baseline.TAU_US,
                   "ts_us": baseline.TS_US, "init": baseline.INIT, "seed": baseline.SEED},
        "design_metrics": m,
        "rows": rows,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [f"# 采样周期误差扫描（约束 {args.depth:g} dB，单配置）", "",
             f"设计对：B = {baseline.B_HZ / 1e6:g} MHz、L = {baseline.N}、"
             f"t_0 = {baseline.T0_US:g} μs、T_L = {baseline.TAU_US:g} μs、"
             f"T_s = {baseline.TS_US:g} μs；约束 {args.depth:g} dB；random_phase、seed {baseline.SEED}。",
             f"设计对指标：PSL {m['psl_db']:.2f} dB、损失 {m['processing_gain_loss_db']:.2f} dB、"
             f"零延迟 {m['isrj_zero_delay_db']:.2f} dB。",
             f"误差范围 ±{args.err_percent:g}% × T_s = ±{span:g} μs，步长 {args.step_us:g} μs。", "",
             "| ΔT_s (μs) | 实际 T_s (μs) | 零延迟电平 (dB) | 幅度比 | 抬升 (dB) | IL |",
             "|---|---|---|---|---|---|"]
    for r in rows:
        if abs(r["ts_error_us"] * 5 - round(r["ts_error_us"] * 5)) > 1e-6:
            continue                       # 表里每 0.2 μs 取一行
        lines.append(f"| {r['ts_error_us']:+.2f} | {r['ts_us']:.2f} | "
                     f"{r['zero_delay_db']:.2f} | {r['amp_ratio']:.3e} | "
                     f"{r['rise_db']:+.2f} | {r['il']:.3e} |")
    (OUTDIR / f"结果表_{abs(args.depth):g}dB.md").write_text("\n".join(lines) + "\n",
                                                             encoding="utf-8")
    print("  JSON 与结果表已保存")


if __name__ == "__main__":
    main()
