"""约束深度对比（沿采样周期维度）：−80 dB vs −30 dB，其余参数完全一致。

口径：搜索线基线 v2（`scripts/baseline.py`，B = 10 MHz、L = 256、t_0 = 2 μs、
T_L = 1.5 μs、T_s = 6.4 μs），**只把零延迟干扰约束 `peak_interf_db` 从 −80 dB 换成 −30 dB**，
两个设计各自用固定设计对（录波口径）扫实际采样周期误差：

    y_j(θ*) = x ⊙ j(θ*)，  z = 20lg( |h^H y_j(θ*)| / |h^H x| )

矩阵链（零延迟原位，未变）：J(θ) = diag(j(θ))、H_j = H·diag(j(θ))、X_j 由 x⊙j 铺成。

扫描：ΔT_s ∈ ±0.5 μs（论文 Fig.12 的误差区间），步长 0.05 μs（= 半个采样点）。

用法：
    .venv/Scripts/python.exe 参数搜索/scripts/run_depth_compare_ts.py
"""

from __future__ import annotations

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

DEPTHS = [-80.0, -30.0]
ERR = np.arange(-0.5, 0.5001, 0.05)
DESIGN_DIR = ROOT / "参数搜索" / "结果" / "设计对"
OUTDIR = ROOT / "参数搜索" / "结果" / "约束深度对比"
COLORS = {-80.0: "#2a78d6", -30.0: "#eb6834"}
TEXT_PRIMARY, TEXT_SECONDARY = "#0b0b0b", "#52514e"
SURFACE, GRID_C = "#fcfcfb", "#d8d7d2"


def design(depth: float) -> tuple:
    npz = DESIGN_DIR / f"joint_design_v2_t0_2us_bmin{abs(depth):g}dB.npz"
    if npz.exists():
        d = np.load(npz)
        print(f"  加载设计对：{npz.name}")
        return d["x"], d["h"], {"psl_db": compute_metrics(d["x"], d["h"], d["j"])["psl_db"],
                                "processing_gain_loss_db":
                                    compute_metrics(d["x"], d["h"], d["j"])["processing_gain_loss_db"],
                                "isrj_zero_delay_db":
                                    compute_metrics(d["x"], d["h"], d["j"])["isrj_zero_delay_db"],
                                "elapsed_s": float("nan")}
    print(f"  联合设计（约束 {depth:g} dB）...")
    DESIGN_DIR.mkdir(parents=True, exist_ok=True)
    res = joint_design_algorithm(baseline.params_for(peak_interf_db=depth), verbose=False)
    np.savez_compressed(npz, x=res["x"], h=res["h"], j=res["j"])
    print(f"    {res['elapsed_s']:.1f} s，PSL {res['psl_db']:.2f} dB，"
          f"零延迟 {res['isrj_zero_delay_db']:.2f} dB")
    return res["x"], res["h"], res


def sweep(x, h, depth: float):
    peak = abs(np.vdot(h, x))
    z = []
    for e in ERR:
        p = baseline.params_for(ts_us=baseline.TS_US + float(e), peak_interf_db=depth)
        r = cross_correlation(x * generate_isrj_sampling_function(p), h)
        z.append(20 * np.log10(abs(r[baseline.N - 1]) / peak + 1e-30))
    return np.array(z)


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
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    OUTDIR.mkdir(parents=True, exist_ok=True)
    print("=== 约束深度对比（沿采样周期维度）===")
    print("  " + baseline.summary().replace(chr(10), chr(10) + "  "))

    curves, metrics = {}, {}
    for depth in DEPTHS:
        x, h, res = design(depth)
        z = sweep(x, h, depth)
        curves[depth] = z
        i0 = int(np.argmin(np.abs(ERR)))
        idx = {e: int(np.argmin(np.abs(ERR - e))) for e in (0.05, 0.1, 0.25, 0.5)}
        metrics[depth] = {
            "valley_db": float(z[i0]),
            "rise_0p1": float(z[idx[0.1]] - z[i0]),
            "rise_0p25": float(z[idx[0.25]] - z[i0]),
            "rise_0p5": float(z[idx[0.5]] - z[i0]),
            "z_0p1": float(z[idx[0.1]]), "z_0p5": float(z[idx[0.5]]),
            "psl_db": None if res is None else float(res["psl_db"]),
            "loss_db": None if res is None else float(res["processing_gain_loss_db"]),
        }
        m = metrics[depth]
        print(f"  约束 {depth:g} dB：谷底 {m['valley_db']:.2f} dB；"
              f"±0.1 μs 抬升 {m['rise_0p1']:.2f} dB（电平 {m['z_0p1']:.2f}）；"
              f"±0.5 μs 抬升 {m['rise_0p5']:.2f} dB（电平 {m['z_0p5']:.2f}）")

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8), facecolor=SURFACE)
    ax = axes[0]
    for depth in DEPTHS:
        ax.plot(ERR, curves[depth], color=COLORS[depth], linewidth=2, marker="o",
                markersize=3.5, label=f"design at {depth:g} dB")
    ax.axhline(-30, color=TEXT_SECONDARY, linewidth=1.0, linestyle="-.")
    ax.axvline(0.0, color=TEXT_SECONDARY, linewidth=1.0, linestyle="--", alpha=0.8)
    ax.set_xlabel("Sampling-period error (μs)")
    ax.set_ylabel("Normalized level after filtering (dB)")
    ax.set_title("(a) Absolute level along the sampling-period axis", fontsize=11)
    ax.legend(frameon=False, fontsize=9, labelcolor=TEXT_SECONDARY)
    style(ax)

    ax = axes[1]
    for depth in DEPTHS:
        ax.plot(ERR, curves[depth] - curves[depth][len(ERR) // 2], color=COLORS[depth],
                linewidth=2, marker="o", markersize=3.5,
                label=f"design at {depth:g} dB")
    ax.axvline(0.0, color=TEXT_SECONDARY, linewidth=1.0, linestyle="--", alpha=0.8)
    ax.set_xlabel("Sampling-period error (μs)")
    ax.set_ylabel("Rise above own valley (dB)")
    ax.set_title("(b) Same data, rise relative to each own valley", fontsize=11)
    ax.legend(frameon=False, fontsize=9, labelcolor=TEXT_SECONDARY)
    style(ax)

    fig.tight_layout()
    out = OUTDIR / "约束深度对比_采样周期.png"
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"  图已保存：{out.name}")

    (OUTDIR / "约束深度对比.json").write_text(json.dumps({
        "baseline": {"B_hz": baseline.B_HZ, "N": baseline.N, "T_p_s": baseline.T_P_S,
                     "t0_us": baseline.T0_US, "tau_us": baseline.TAU_US,
                     "ts_us": baseline.TS_US, "err_us": ERR.tolist()},
        "curves": {f"{d:g}dB": curves[d].tolist() for d in DEPTHS},
        "metrics": {f"{d:g}dB": metrics[d] for d in DEPTHS},
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = ["# 约束深度对比（沿采样周期维度；其余参数为 v2 基线）", "",
             f"设计对参数：B = {baseline.B_HZ / 1e6:g} MHz、L = {baseline.N}、"
             f"t_0 = {baseline.T0_US:g} μs、T_L = {baseline.TAU_US:g} μs、"
             f"T_s = {baseline.TS_US:g} μs；只改零延迟干扰约束。",
             f"扫描 ΔT_s ∈ ±0.5 μs（步长 0.05 μs = 半个采样点），固定设计对（录波）口径。", "",
             "| 约束 | 谷底 (dB) | ±0.1 μs | ±0.25 μs | ±0.5 μs | ±0.5 μs 绝对电平 (dB) | PSL (dB) | 损失 (dB) |",
             "|---|---|---|---|---|---|---|---|"]
    for depth in DEPTHS:
        m = metrics[depth]
        psl = "—" if m["psl_db"] is None else f"{m['psl_db']:.2f}"
        loss = "—" if m["loss_db"] is None else f"{m['loss_db']:.2f}"
        lines.append(f"| {depth:g} dB | {m['valley_db']:.2f} | +{m['rise_0p1']:.2f} dB | "
                     f"+{m['rise_0p25']:.2f} dB | +{m['rise_0p5']:.2f} dB | {m['z_0p5']:.2f} | "
                     f"{psl} | {loss} |")
    (OUTDIR / "结果表.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("  JSON 与结果表已保存")


if __name__ == "__main__":
    main()
