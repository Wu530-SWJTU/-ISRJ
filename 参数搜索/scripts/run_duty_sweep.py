"""占空比误差一维扫描（单配置）：归一化电平 – Δd。

口径：搜索线基线 v2（`scripts/baseline.py`），**固定设计对**（录波口径）。
占空比 `d = T_L/T_s` 是派生量；固定 `T_s` = 6.4 μs 时，占空比误差等价于截取长度误差：

    T_L_actual = (d + Δd) · T_s      ⇔      ΔT_L = Δd · T_s

取点步长 = **1 个采样点在 `d` 上的等价量** `1/(T_s·fs)` = 1/64 ≈ 0.0156，范围 ±5 步
（即 ΔT_L ∈ ±0.5 μs，与三对联合误差图的范围一致）。设计对、约束（−80 dB）、
零延迟原位模型与其它图完全一致。

矩阵链（未变）：J(θ) = diag(j(θ))、H_j = H·diag(j(θ))、X_j 由 x⊙j 铺成。

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

D0 = baseline.TAU_US / baseline.TS_US                 # 基线占空比 0.234
STEP_D = baseline.SAMPLE_US / baseline.TS_US          # 1 个采样点的等价 d 步长 = 1/64
DESIGN_DIR = ROOT / "参数搜索" / "结果" / "参数调制" / "设计对"
OUTDIR = ROOT / "参数搜索" / "结果" / "参数调制" / "占空比扫描"
SERIES_1 = "#2a78d6"
TEXT_PRIMARY, TEXT_SECONDARY = "#0b0b0b", "#52514e"
SURFACE, GRID_C = "#fcfcfb", "#d8d7d2"


def design(depth: float) -> tuple:
    npz = DESIGN_DIR / f"joint_design_v2_t0_2us_bmin{abs(depth):g}dB.npz"
    if npz.exists():
        d = np.load(npz)
        print(f"  加载设计对：{npz.name}")
        return d["x"], d["h"], d["j"], None
    print("  设计对不存在，跑一次联合设计 ...")
    DESIGN_DIR.mkdir(parents=True, exist_ok=True)
    res = joint_design_algorithm(baseline.params_for(peak_interf_db=depth), verbose=False)
    np.savez_compressed(npz, x=res["x"], h=res["h"], j=res["j"])
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
    ap.add_argument("--steps", type=int, default=5,
                    help="单侧步数（默认 5 ⇒ ΔT_L = ±0.5 μs = ±5 个采样点）")
    ap.add_argument("--depth", type=float, default=baseline.PEAK_INTERF_DB,
                    help="零延迟干扰约束 [dB]（默认跟随基线）")
    ap.add_argument("--annotate", action="store_true",
                    help="叠加标记点与峰值虚线（默认纯折线）")
    args = ap.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    OUTDIR.mkdir(parents=True, exist_ok=True)

    dd = np.arange(-args.steps, args.steps + 1) * STEP_D      # Δd 网格
    print(f"=== 占空比误差一维扫描（约束 {args.depth:g} dB）===")
    print("  " + baseline.summary().replace(chr(10), chr(10) + "  "))
    print(f"  基线占空比 d = {D0:.4f}；Δd 步长 {STEP_D:.6f}（= 1 个采样点）；"
          f"范围 ±{args.steps * STEP_D:.4f}（ΔT_L = ±{args.steps * baseline.SAMPLE_US:g} μs）")

    x, h, j, res = design(args.depth)
    m = compute_metrics(x, h, j)
    peak = abs(np.vdot(h, x))
    print(f"  设计对指标：PSL {m['psl_db']:.2f} dB、损失 {m['processing_gain_loss_db']:.2f} dB、"
          f"零延迟 {m['isrj_zero_delay_db']:.2f} dB")

    rows = []
    for e in dd:
        d_actual = D0 + float(e)
        p = baseline.params_for(tau_us=d_actual * baseline.TS_US, peak_interf_db=args.depth)
        r = cross_correlation(x * generate_isrj_sampling_function(p), h)
        zero = float(20 * np.log10(abs(r[baseline.N - 1]) / peak + 1e-30))
        rows.append({"d_error": round(float(e), 8), "d": round(d_actual, 6),
                     "tau_us": round(d_actual * baseline.TS_US, 6),
                     "zero_delay_db": zero,
                     "peak_db": float(20 * np.log10(np.abs(r).max() / peak + 1e-30)),
                     "amp_ratio": float(10 ** (zero / 20)),
                     "rise_db": float(zero - m["isrj_zero_delay_db"]),
                     "il": float(np.sum(np.abs(r) ** 2))})
    i0 = len(rows) // 2
    print(f"  谷底（Δd=0）：{rows[i0]['zero_delay_db']:.2f} dB；"
          f"两端 Δd=±{args.steps * STEP_D:.4f}（T_L={rows[0]['tau_us']:g}/"
          f"{rows[-1]['tau_us']:g} μs）：{rows[0]['zero_delay_db']:.2f} / "
          f"{rows[-1]['zero_delay_db']:.2f} dB")

    e = np.array([r["d_error"] for r in rows])
    fig, ax = plt.subplots(1, 1, figsize=(7.2, 4.7), facecolor=SURFACE)

    ax.plot(e, [r["zero_delay_db"] for r in rows], color=SERIES_1, linewidth=2,
            marker="o", markersize=4, label="zero-delay level")
    ax.axhline(args.depth, color=TEXT_SECONDARY, linewidth=1.0,
               linestyle="-.")
    ax.axvline(0.0, color=TEXT_SECONDARY, linewidth=1.0, linestyle="--", alpha=0.8)
    ax.set_xlabel("Duty-cycle error Δd")
    ax.set_ylabel("Normalized level (dB)")
    ax.set_title(f"Level after filtering (design constraint {args.depth:g} dB)",
                 fontsize=11)
    ax.annotate(f"$d$ = {D0:.4f} ($T_L$ = {baseline.TAU_US:g} μs) at zero error; "
                f"valley = {rows[i0]['zero_delay_db']:.2f} dB",
                xy=(0.03, 0.06), xycoords="axes fraction", fontsize=9, color=TEXT_PRIMARY)
    ax.legend(frameon=False, fontsize=9, labelcolor=TEXT_SECONDARY)
    style(ax)

    fig.tight_layout()
    tag = "标注版" if args.annotate else "折线"
    out = OUTDIR / f"占空比误差扫描_约束{abs(args.depth):g}dB_{tag}.png"
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"  图已保存：{out.name}")

    (OUTDIR / f"duty_sweep_{abs(args.depth):g}dB.json").write_text(json.dumps({
        "config": {"constraint_db": args.depth, "d0": D0,
                   "step_d": STEP_D, "steps": args.steps,
                   "B_hz": baseline.B_HZ, "N": baseline.N, "T_p_s": baseline.T_P_S,
                   "t0_us": baseline.T0_US, "tau_us": baseline.TAU_US,
                   "ts_us": baseline.TS_US, "init": baseline.INIT, "seed": baseline.SEED},
        "design_metrics": m, "rows": rows,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [f"# 占空比误差一维扫描（约束 {args.depth:g} dB，单配置）", "",
             f"设计对：B = {baseline.B_HZ / 1e6:g} MHz、L = {baseline.N}、"
             f"t_0 = {baseline.T0_US:g} μs、T_L = {baseline.TAU_US:g} μs、"
             f"T_s = {baseline.TS_US:g} μs（d = {D0:.4f}）；"
             f"约束 {args.depth:g} dB；random_phase、seed {baseline.SEED}。",
             f"设计对指标：PSL {m['psl_db']:.2f} dB、损失 {m['processing_gain_loss_db']:.2f} dB、"
             f"零延迟 {m['isrj_zero_delay_db']:.2f} dB。",
             f"取点步长 Δd = {STEP_D:.6f}（= 1 个采样点 = 0.1 μs / T_s），"
             f"范围 ±{args.steps * STEP_D:.4f}（ΔT_L = ±{args.steps * baseline.SAMPLE_US:g} μs）。", "",
             "| Δd | 实际 d | T_L (μs) | 零延迟电平 (dB) | 幅度比 | 抬升 (dB) | IL |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['d_error']:+.4f} | {r['d']:.4f} | {r['tau_us']:.2f} | "
                     f"{r['zero_delay_db']:.2f} | {r['amp_ratio']:.3e} | "
                     f"{r['rise_db']:+.2f} | {r['il']:.3e} |")
    (OUTDIR / f"结果表_{abs(args.depth):g}dB.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("  JSON 与结果表已保存")


if __name__ == "__main__":
    main()
