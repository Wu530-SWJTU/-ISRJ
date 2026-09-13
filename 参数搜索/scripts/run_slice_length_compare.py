"""截取长度对比：缩短设计端的 T_L，看占空比/截取长度维度能否出现 V 型。

背景：T_L = 2.5 μs 的设计对，占空比误差曲线是单调上升（左端更"好"），原因是
实际窗口变短时采样点数变少、干扰总能量变小 ⇒ 绝对电平自然下降。本实验把设计端
的 T_L 缩短（2.5 / 1.0 / 0.5 μs），看：
  ① 绝对口径（相对主瓣）是否出现极小点；
  ② 相对口径（相对干扰自身匹配峰 Σj）的 V 型是否更清晰。

口径：固定设计对（录波口径），扫**实际**窗口宽度 T_L + ΔT_L：
    y_j(θ*) = x ⊙ j(θ*)，  θ* = (t_0, T_L + ΔT_L, T_s)
    z_abs = 20lg( |h^H y_j| / |h^H x| )          （分母 = b_max = N）
    z_rel = 20lg( |h^H y_j| / Σ j )              （分母 = 干扰自身匹配峰 = 采样点数）

矩阵链（未变）：J(θ) = diag(j(θ))、H_j = H·diag(j(θ))、X_j 由 x⊙j 铺成；
本参数下只有 `N_tau` 随宽度变化。

横轴用**绝对扰动** ΔT_L（μs，±0.5、步长 0.05）而非 Δd：三个设计的 d 不同，
同一 ΔT_L 的绝对扰动相同才可比（Δd = ΔT_L / T_s = ΔT_L/10）。

用法：
    .venv/Scripts/python.exe 参数搜索/scripts/run_slice_length_compare.py
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

from isrj_joint_design.joint_design import compute_metrics, joint_design_algorithm  # noqa: E402
from isrj_joint_design.matrices import cross_correlation  # noqa: E402
from isrj_joint_design.params import GlobalParams, generate_isrj_sampling_function  # noqa: E402

N = 800
T0_US, TS_US = 2.5, 10.0
DESIGNS = [2.5, 1.0, 0.5]                 # 设计端截取长度 [μs]
DELTA = np.arange(-0.45, 0.5001, 0.05)    # 实际窗口宽度扰动 [μs]（> -T_L_min 保证非零）
OUTDIR = ROOT / "参数搜索" / "结果" / "截取长度对比"
COLORS = {2.5: "#2a78d6", 1.0: "#eb6834", 0.5: "#3f9142"}
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
SURFACE = "#fcfcfb"
GRID = "#d8d7d2"


def params_for(tl_us: float) -> GlobalParams:
    return GlobalParams(N=N, tau=tl_us * 1e-6, T_s=TS_US * 1e-6,
                        tau_offset=T0_US * 1e-6, seed=0)


def design(tl_us: float) -> tuple:
    npz = OUTDIR / f"joint_design_TL{tl_us:g}us.npz"
    if npz.exists():
        d = np.load(npz)
        print(f"  加载已有设计：{npz.name}")
        return d["x"], d["h"], None
    print(f"  联合设计（T_L = {tl_us:g} μs，d = {tl_us / TS_US:.2f}）...")
    res = joint_design_algorithm(params_for(tl_us), verbose=False)
    np.savez_compressed(npz, x=res["x"], h=res["h"], j=res["j"])
    print(f"    {res['elapsed_s']:.1f} s，PSL={res['psl_db']:.2f} dB，"
          f"损失={res['processing_gain_loss_db']:.2f} dB，"
          f"残差=({res['residual_b_max']:.1e}, {res['residual_b_min']:.1e})")
    return res["x"], res["h"], res


def sweep(x: np.ndarray, h: np.ndarray, tl_us: float) -> tuple:
    peak = abs(np.vdot(h, x))
    abs_db, rel_db, counts = [], [], []
    for dl in DELTA:
        tl = tl_us + float(dl)
        if tl <= 0.01:
            abs_db.append(np.nan); rel_db.append(np.nan); counts.append(0.0)
            continue
        j = generate_isrj_sampling_function(params_for(tl))
        r = cross_correlation(x * j, h)
        n_samp = float(np.sum(j))
        abs_db.append(float(20 * np.log10(abs(r[N - 1]) / peak + 1e-30)))
        rel_db.append(float(20 * np.log10(abs(r[N - 1]) / max(n_samp, 1e-30) + 1e-30)))
        counts.append(n_samp)
    return np.array(abs_db), np.array(rel_db), np.array(counts)


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


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    OUTDIR.mkdir(parents=True, exist_ok=True)

    print("=== 截取长度对比（设计端 T_L 缩短，看能否出 V）===")
    curves, rows = {}, []
    for tl in DESIGNS:
        x, h, res = design(tl)
        m = compute_metrics(x, h, np.load(OUTDIR / f"joint_design_TL{tl:g}us.npz")["j"])
        a, rl, cnt = sweep(x, h, tl)
        curves[tl] = (a, rl, cnt)
        i0 = int(np.argmin(np.abs(DELTA)))
        rows.append({
            "TL_design_us": tl, "d_design": tl / TS_US,
            "psl_db": m["psl_db"], "gain_loss_db": m["processing_gain_loss_db"],
            "abs_at_design": float(a[i0]), "rel_at_design": float(rl[i0]),
            "abs_min": float(np.nanmin(a)), "abs_argmin_us": float(DELTA[int(np.nanargmin(a))]),
            "rel_min": float(np.nanmin(rl)), "rel_argmin_us": float(DELTA[int(np.nanargmin(rl))]),
        })
        q = rows[-1]
        print(f"  T_L = {tl:g} μs（d = {q['d_design']:.2f}）：PSL {q['psl_db']:.2f} dB、"
              f"损失 {q['gain_loss_db']:.2f} dB")
        print(f"    绝对口径：设计点 {q['abs_at_design']:.2f} dB，全段最小 {q['abs_min']:.2f} dB "
              f"在 ΔT_L = {q['abs_argmin_us']:+.2f} μs"
              f"{'  ← 极小不在设计点' if abs(q['abs_argmin_us']) > 1e-9 else '  ← 极小在设计点'}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), facecolor=SURFACE)
    for tl in DESIGNS:
        a, rl, _ = curves[tl]
        q = next(rr for rr in rows if rr["TL_design_us"] == tl)
        axes[0].plot(DELTA, a, color=COLORS[tl], linewidth=2, marker="o", markersize=3.5,
                     label=f"T_L = {tl:g} μs (d = {q['d_design']:.2f}), PSL {q['psl_db']:.1f} dB")
        axes[1].plot(DELTA, rl, color=COLORS[tl], linewidth=2, marker="o", markersize=3.5,
                     label=f"T_L = {tl:g} μs")
    for ax, ttl, ylb in ((axes[0], "(a) Level relative to mainlobe", "Normalized level (dB)"),
                         (axes[1], "(b) Suppression relative to jammer's own matched peak",
                          "|h^H y_j| / Σ j  (dB)")):
        ax.axvline(0.0, color=TEXT_SECONDARY, linewidth=1.0, linestyle="--", alpha=0.8)
        ax.set_xlabel("Slice-length error ΔT_L (μs)")
        ax.set_ylabel(ylb)
        ax.set_title(ttl, fontsize=11)
        ax.legend(frameon=False, fontsize=9, labelcolor=TEXT_SECONDARY)
        _style(ax)
    fig.tight_layout()
    out = OUTDIR / "截取长度对比_两种口径.png"
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"  图已保存：{out.name}")

    (OUTDIR / "截取长度对比.json").write_text(json.dumps({
        "delta_tl_us": DELTA.tolist(),
        "curves": {f"{tl:g}": {"abs_db": curves[tl][0].tolist(),
                               "rel_db": curves[tl][1].tolist(),
                               "jamming_samples": curves[tl][2].tolist()}
                   for tl in DESIGNS},
        "summary": rows,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = ["# 截取长度对比（设计端 T_L 缩短）", "",
             f"扫描实际窗口宽度扰动 ΔT_L ∈ [{DELTA[0]:g}, {DELTA[-1]:g}] μs，步长 0.05；"
             f"t_0 = {T0_US:g} μs、T_s = {TS_US:g} μs。", "",
             "| 设计 T_L (μs) | d | PSL (dB) | 损失 (dB) | 绝对口径@设计点 | 绝对口径最小值 (位置) | 相对口径@设计点 |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['TL_design_us']:g} | {r['d_design']:.2f} | {r['psl_db']:.2f} | "
                     f"{r['gain_loss_db']:.2f} | {r['abs_at_design']:.2f} | "
                     f"{r['abs_min']:.2f} ({r['abs_argmin_us']:+.2f} μs) | {r['rel_at_design']:.2f} |")
    (OUTDIR / "结果表.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("  JSON 与结果表已保存")


if __name__ == "__main__":
    main()
