"""码内采样窗口数对比：4 个窗口（T_s = 10 μs）vs 8 个窗口（T_s = 5 μs）。

问题：为什么 V 型平缓？机理——零延迟泄漏 = Σ_{位移样本} ±conj(h[n])·x[n]，
每单位误差的位移样本数 = 2·ε·fs·K（K = 码内窗口数）。K 越大，同样误差扰动越多
样本 ⇒ V 越陡。本实验只改 T_s（窗口数），其余不变：

    配置 A（现状）：T_s = 10 μs → 200 点/周期 → K = 4，占空比 0.25
    配置 B（本次）：T_s =  5 μs → 100 点/周期 → K = 8，占空比 0.50

两配置的 T_L = 2.5 μs（50 点）、t_0 = T_s/4（A: 2.5 μs、B: 1.25 μs，相对位置相同，
保证窗口不跨周期：t_0 + T_L ≤ T_s）。扫描 ε ∈ ±1.25 μs（B 的对称可用范围），
步长 0.05 μs（1 个采样点）。

干扰模型与矩阵链（未变，只换 T_s）：
    信号级：j(θ)[n] = 1 ⟺ mod(n, N_Ts) ∈ [N_tau_offset, N_tau_offset + N_tau)
            y_j(θ) = x ⊙ j(θ)
    矩阵级：J(θ) = diag(j(θ))，H_j = H·diag(j(θ))；X_j 由 x⊙j 铺成
    配置 B：N_Ts = 100、N_tau = 50、N_tau_offset = 25 → 每周期 [25, 75) 点

用法：
    .venv/Scripts/python.exe 参数搜索/scripts/run_window_count_compare.py
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
TAU_US = 2.5                       # 截取长度（两配置相同）
CONFIGS = {                        # 名称: T_s [μs]
    "K=4 (Ts=10us)": 10.0,
    "K=8 (Ts=5us)": 5.0,
}
ERR = np.arange(-1.25, 1.2501, 0.05)   # 误差网格 [μs]，步长 = 1 个采样点
OUTDIR = ROOT / "参数搜索" / "结果" / "采样周期对比"
COLORS = {"K=4 (Ts=10us)": "#2a78d6", "K=8 (Ts=5us)": "#eb6834"}
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
SURFACE = "#fcfcfb"
GRID = "#d8d7d2"


def params_for(t_s_us: float, t0_us: float) -> GlobalParams:
    return GlobalParams(N=N, tau=TAU_US * 1e-6, T_s=t_s_us * 1e-6,
                        tau_offset=t0_us * 1e-6, seed=0)


def design(t_s_us: float) -> tuple:
    t0 = t_s_us / 4.0
    npz = OUTDIR / f"joint_design_Ts{t_s_us:g}us_t0{t0:g}us.npz"
    if npz.exists():
        d = np.load(npz)
        print(f"  加载已有设计：{npz.name}")
        return d["x"], d["h"], d["j"], None
    print(f"  联合设计（T_s = {t_s_us:g} μs、t_0 = {t0:g} μs）...")
    res = joint_design_algorithm(params_for(t_s_us, t0), verbose=False)
    np.savez_compressed(npz, x=res["x"], h=res["h"], j=res["j"])
    print(f"    {res['elapsed_s']:.1f} s，PSL={res['psl_db']:.2f} dB，"
          f"损失={res['processing_gain_loss_db']:.2f} dB，"
          f"残差=({res['residual_b_max']:.1e}, {res['residual_b_min']:.1e})")
    return res["x"], res["h"], res["j"], res


def sweep(x: np.ndarray, h: np.ndarray, t_s_us: float) -> np.ndarray:
    peak = abs(np.vdot(h, x))
    out = []
    for e in ERR:
        ja = generate_isrj_sampling_function(params_for(t_s_us, t_s_us / 4.0 + float(e)))
        r = cross_correlation(x * ja, h)
        out.append(20 * np.log10(abs(r[(len(r) - 1) // 2]) / peak + 1e-30))
    return np.array(out)


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

    print("=== 码内采样窗口数对比（T_L = 2.5 μs 不变，只改 T_s）===")
    curves, rows = {}, []
    for name, ts in CONFIGS.items():
        x, h, j, res = design(ts)
        m = compute_metrics(x, h, j)
        z = sweep(x, h, ts)
        curves[name] = z
        i0 = int(np.argmin(np.abs(ERR)))
        idx = {e: int(np.argmin(np.abs(ERR - e))) for e in (0.05, 0.1, 0.25, 0.5, 1.0)}
        rows.append({
            "config": name, "ts_us": ts, "windows": int(round(N / (ts * 20))),
            "duty": TAU_US / ts, "t0_us": ts / 4.0,
            "psl_db": m["psl_db"], "gain_loss_db": m["processing_gain_loss_db"],
            "valley_db": float(z[i0]),
            "z_0p05": float(z[idx[0.05]]), "rise_0p05": float(z[idx[0.05]] - z[i0]),
            "z_0p25": float(z[idx[0.25]]), "rise_0p25": float(z[idx[0.25]] - z[i0]),
            "z_0p5": float(z[idx[0.5]]), "rise_0p5": float(z[idx[0.5]] - z[i0]),
            "z_1p0": float(z[idx[1.0]]), "rise_1p0": float(z[idx[1.0]] - z[i0]),
        })
        r = rows[-1]
        print(f"  {name}：窗口 {r['windows']} 个、占空比 {r['duty']:.2f}、"
              f"PSL {r['psl_db']:.2f} dB、损失 {r['gain_loss_db']:.2f} dB")
        print(f"    谷底 {r['valley_db']:.2f} dB；±0.05μs 抬升 {r['rise_0p05']:5.2f} dB；"
              f"±0.25μs {r['rise_0p25']:5.2f} dB；±0.5μs {r['rise_0p5']:5.2f} dB；"
              f"±1.0μs {r['rise_1p0']:5.2f} dB")

    fig, ax = plt.subplots(figsize=(7.8, 5.0), facecolor=SURFACE)
    for name in CONFIGS:
        r = next(rr for rr in rows if rr["config"] == name)
        ax.plot(ERR, curves[name], color=COLORS[name], linewidth=2, marker="o",
                markersize=3.5,
                label=f"{name}: {r['windows']} windows, duty {r['duty']:.2f}, "
                      f"PSL {r['psl_db']:.1f} dB")
    ax.axhline(-30, color=TEXT_SECONDARY, linewidth=1.0, linestyle="-.")
    ax.axvline(0.0, color=TEXT_SECONDARY, linewidth=1.0, linestyle="--", alpha=0.8)
    ax.set_xlabel("Window-start error (μs)")
    ax.set_ylabel("Normalized ISRJ level after filtering (dB)")
    ax.set_title("Effect of sampling-window count on the error response "
                 "(T_L = 2.5 μs fixed)", fontsize=11)
    ax.legend(frameon=False, fontsize=9, labelcolor=TEXT_SECONDARY)
    _style(ax)
    fig.tight_layout()
    fig.savefig(OUTDIR / "窗口数对比_归一化电平.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"  图已保存：{(OUTDIR / '窗口数对比_归一化电平.png').name}")

    (OUTDIR / "窗口数对比.json").write_text(json.dumps(
        {"errors_us": ERR.tolist(),
         "curves": {k: v.tolist() for k, v in curves.items()},
         "summary": rows}, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = ["# 码内采样窗口数对比（T_L = 2.5 μs 不变，只改 T_s）", "",
             f"扫描 ε ∈ ±1.25 μs（步长 0.05 μs = 1 个采样点）；t_0 基线 = T_s/4。", "",
             "| 配置 | 窗口数 | 占空比 | PSL (dB) | 损失 (dB) | 谷底 (dB) | ±0.05μs | ±0.25μs | ±0.5μs | ±1.0μs |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['config']} | {r['windows']} | {r['duty']:.2f} | "
                     f"{r['psl_db']:.2f} | {r['gain_loss_db']:.2f} | {r['valley_db']:.2f} | "
                     f"{r['rise_0p05']:.2f} | {r['rise_0p25']:.2f} | "
                     f"{r['rise_0p5']:.2f} | {r['rise_1p0']:.2f} |")
    lines += ["", "（表内后四列为相对各自谷底的**抬升量**，dB，越大越尖。）"]
    (OUTDIR / "结果表.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  结果表已保存：结果表.md")


if __name__ == "__main__":
    main()
