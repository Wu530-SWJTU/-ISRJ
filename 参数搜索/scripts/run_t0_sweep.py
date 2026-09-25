"""窗口起点 t_0 估计误差扫描：归一化电平 – t_0（搜索线 v2 · 对齐 Zhou 2026）。

设计对来自基线单一来源 `scripts/baseline.py`：
    B = 10 MHz、T_p = 25.6 μs、码长 L = 256、fs = B ⇒ 1 采样点 = 0.1 μs
    t_0 = 2 μs（20 点）、T_L = 1.5 μs（15 点）、T_s = 6.4 μs（64 点/周期，码内 4 片）
    占空比 d = 0.234、零延迟干扰约束 b_min = b_max·10⁻⁴（−80 dB）

实际窗口起点取 t_0 + ε（默认扫 ±1.0 μs；论文 Fig.12 的误差区间是 ±0.5 μs）。

干扰模型（**零延迟原位**，与复现线共用公式，只换窗口起点）：
    信号级：j(θ)[n] = 1 ⟺ mod(n, N_Ts) ∈ [N_tau_offset, N_tau_offset + N_tau)
            y_j(θ) = x ⊙ j(θ)
    矩阵级：J(θ) = diag(j(θ))，H_j = H·diag(j(θ))，X_j 由 x⊙j 铺成

归一化分母：|h^H x| = b_max = N（主瓣硬约束钉死，不随误差变化）
搜索线代价：C = 10^(isrj_peak_db / 20)

用法：
    .venv/Scripts/python.exe 参数搜索/scripts/run_t0_sweep.py
    .venv/Scripts/python.exe 参数搜索/scripts/run_t0_sweep.py --redesign   # 强制重跑设计
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
from baseline import N, SEED, T0_US as T0_DESIGN_US, TAU_US, TS_US  # noqa: E402
from isrj_joint_design.joint_design import compute_metrics, joint_design_algorithm  # noqa: E402
from isrj_joint_design.matrices import cross_correlation  # noqa: E402
from isrj_joint_design.params import generate_isrj_sampling_function  # noqa: E402

OUTDIR = ROOT / "参数搜索" / "结果" / "窗口起点"
DESIGN_NPZ = OUTDIR / "joint_design_v2_t0_2us.npz"

# 与复现线一致的绘图配色（plotting.py 的 SERIES_1/2）
SERIES_1 = "#2a78d6"
SERIES_2 = "#eb6834"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
SURFACE = "#fcfcfb"
GRID = "#d8d7d2"


def params_for(t0_us: float):
    """给定窗口起点构造参数（其余取搜索线基线 v2，见 baseline.py）。"""
    return baseline.params_for(t0_us=t0_us)


def design(force: bool) -> tuple:
    """基线 v2 的联合设计（有存档则直接加载）。"""
    if DESIGN_NPZ.exists() and not force:
        d = np.load(DESIGN_NPZ)
        print(f"  加载已有设计：{DESIGN_NPZ.name}")
        return d["x"], d["h"], d["j"], None

    print(f"  联合设计（t_0 = {T0_DESIGN_US:g} μs，N={N}，M=100）...")
    result = joint_design_algorithm(params_for(T0_DESIGN_US))
    OUTDIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(DESIGN_NPZ, x=result["x"], h=result["h"], j=result["j"],
                        isl_history=result["isl_history"],
                        il_history=result["il_history"])
    print(f"  设计完成：{result['elapsed_s']:.1f} s，已存 {DESIGN_NPZ.name}")
    return result["x"], result["h"], result["j"], result


def sweep(x: np.ndarray, h: np.ndarray, eps_us: np.ndarray) -> list:
    """扫实际窗口起点，返回每点的归一化电平 / C / IL。"""
    peak = abs(np.vdot(h, x))              # 归一化分母 = |h^H x| = b_max = N
    rows = []
    for e in eps_us:
        t0 = T0_DESIGN_US + float(e)
        j_actual = generate_isrj_sampling_function(params_for(t0))
        r = cross_correlation(x * j_actual, h)
        zero_db = float(20 * np.log10(abs(r[N - 1]) / peak + 1e-30))
        peak_db = float(20 * np.log10(np.abs(r).max() / peak + 1e-30))
        rows.append({
            "t0_us": round(t0, 6),
            "t0_error_us": round(float(e), 6),
            "zero_delay_db": zero_db,
            "peak_db": peak_db,
            "C": float(10 ** (peak_db / 20)),
            "il": float(np.sum(np.abs(r) ** 2)),
        })
    return rows


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


def plot(rows: list, save_path: Path) -> None:
    err = np.array([r["t0_error_us"] for r in rows])      # 横坐标：t_0 误差 ±μs
    zero = np.array([r["zero_delay_db"] for r in rows])
    peak = np.array([r["peak_db"] for r in rows])
    il = np.array([r["il"] for r in rows])

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), facecolor=SURFACE)

    ax = axes[0]
    ax.plot(err, zero, color=SERIES_1, linewidth=2, marker="o", markersize=4,
            label="zero-delay level")
    # 单份原位转发模型下整体峰值恒等于零延迟值（差异 ~1e-15），标记点单独画，
    # 避免两条曲线重叠导致下面那条不可见。
    ax.plot(err, peak, color=SERIES_2, linestyle="none", marker="s", markersize=3,
            label="overall peak (= zero-delay)")
    ax.axhline(baseline.PEAK_INTERF_DB, color=TEXT_SECONDARY, linewidth=1.0,
               linestyle="-.")
    ax.axvline(0.0, color=TEXT_SECONDARY, linewidth=1.0, linestyle="--", alpha=0.8)
    lo = max(-100, np.floor(np.min(zero) / 10) * 10 - 5)
    ax.set_ylim(lo, 5)
    ax.annotate(f"design baseline $t_0$ = {T0_DESIGN_US:g} $\\mu$s (zero error)\n"
                f"zero-delay at baseline = {zero[np.argmin(np.abs(err))]:.2f} dB",
                xy=(0.03, 0.06), xycoords="axes fraction",
                fontsize=9, color=TEXT_PRIMARY)
    ax.set_title("(a) ISRJ level after filtering, under window-start error",
                 fontsize=11)
    ax.set_xlabel("Window-start error (μs)")
    ax.set_ylabel("Normalized level (dB)")
    ax.legend(frameon=False, fontsize=9, labelcolor=TEXT_SECONDARY)
    _style(ax)

    ax = axes[1]
    ax.semilogy(err, il, color=SERIES_1, linewidth=2, marker="o", markersize=4)
    ax.axvline(0.0, color=TEXT_SECONDARY, linewidth=1.0, linestyle="--", alpha=0.8)
    ax.set_title("(b) Integrated jamming level after filtering, under window-start error",
                 fontsize=11)
    ax.set_xlabel("Window-start error (μs)")
    ax.set_ylabel("IL")
    _style(ax)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"  图已保存：{save_path.name}")


def write_table(rows: list, save_path: Path) -> None:
    """每 0.5 μs 取一行，写成可读的结果表。"""
    lines = [
        f"# 归一化电平 – 窗口起点 t_0（设计基线 t_0 = {T0_DESIGN_US:g} μs）",
        "",
        f"设计：N={N}、M=100、init=random_phase、seed={SEED}；"
        f"B = {baseline.B_HZ / 1e6:g} MHz、T_p = {baseline.T_P_S * 1e6:g} μs、"
        f"截取长度 {TAU_US:g} μs、采样周期 {TS_US:g} μs、占空比 "
        f"{TAU_US / TS_US:.2f}。归一化分母 = $|h^H x|$ = b_max = N。",
        "",
        "| 误差 (μs) | 实际 t_0 (μs) | 零延迟电平 (dB) | 整体峰值 (dB) | C | IL |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        if abs(r["t0_error_us"] * 2 - round(r["t0_error_us"] * 2)) > 1e-6:
            continue                     # 只保留 0.5 μs 的整数倍
        lines.append(
            f"| {r['t0_error_us']:+.2f} | {r['t0_us']:.2f} | "
            f"{r['zero_delay_db']:.2f} | {r['peak_db']:.2f} | "
            f"{r['C']:.3f} | {r['il']:.3e} |")
    save_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  结果表已保存：{save_path.name}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--min", type=float, default=-1.0, help="t_0 误差下界 [μs]")
    ap.add_argument("--max", type=float, default=1.0, help="t_0 误差上界 [μs]")
    ap.add_argument("--step", type=float, default=0.1,
                    help="t_0 误差步长 [μs]（1 采样点 = 0.1 μs）")
    ap.add_argument("--redesign", action="store_true", help="忽略存档，重跑联合设计")
    args = ap.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    OUTDIR.mkdir(parents=True, exist_ok=True)

    print("=== 窗口起点 t_0 估计误差扫描（搜索线 v2）===")
    print("  " + baseline.summary().replace(chr(10), chr(10) + "  "))
    x, h, j, result = design(args.redesign)

    if result is not None:
        print(f"  设计指标：ISL={result['final_isl']:.3e}  IL={result['final_il']:.3e}  "
              f"PSL={result['psl_db']:.2f} dB  "
              f"处理增益损失={result['processing_gain_loss_db']:.2f} dB  "
              f"约束残差=({result['residual_b_max']:.1e}, {result['residual_b_min']:.1e})")
    m = compute_metrics(x, h, j)
    print(f"  设计对指标（复算）：PSL={m['psl_db']:.2f} dB  "
          f"ISRJ 零延迟={m['isrj_zero_delay_db']:.2f} dB  "
          f"ISRJ 峰值={m['isrj_peak_db']:.2f} dB")

    eps = np.arange(args.min, args.max + args.step / 2, args.step)
    rows = sweep(x, h, eps)
    i0 = int(np.argmin(np.abs(np.array([r["t0_error_us"] for r in rows]))))
    print(f"  扫描 {len(rows)} 点：ε ∈ [{args.min:g}, {args.max:g}] μs，步长 {args.step:g}")
    print(f"  谷底（ε={rows[i0]['t0_error_us']:+.2f} μs，实际 t_0={rows[i0]['t0_us']:.2f} μs）："
          f"零延迟 {rows[i0]['zero_delay_db']:.2f} dB、C={rows[i0]['C']:.4f}")
    print(f"  两端：ε={rows[0]['t0_error_us']:+.2f} → {rows[0]['zero_delay_db']:.2f} dB ；"
          f"ε={rows[-1]['t0_error_us']:+.2f} → {rows[-1]['zero_delay_db']:.2f} dB")

    plot(rows, OUTDIR / "归一化电平_t0.png")
    write_table(rows, OUTDIR / "结果表.md")
    out = OUTDIR / "t0_sweep.json"
    out.write_text(json.dumps({
        "design": {"t0_us": T0_DESIGN_US, "tau_us": TAU_US, "ts_us": TS_US,
                   "N": N, "seed": SEED, "init": "random_phase",
                   "npz": DESIGN_NPZ.name,
                   "metrics": m},
        "errors_us": [r["t0_error_us"] for r in rows],
        "t0_us": [r["t0_us"] for r in rows],
        "zero_delay_db": [r["zero_delay_db"] for r in rows],
        "peak_db": [r["peak_db"] for r in rows],
        "C": [r["C"] for r in rows],
        "il": [r["il"] for r in rows],
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  JSON 已保存：{out.name}")


if __name__ == "__main__":
    main()
