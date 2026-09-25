"""三对参数联合误差图（v2 口径）：t_0×T_s、t_0×T_L、T_L×T_s。

口径：搜索线基线 v2（`scripts/baseline.py`，对齐 Zhou 2026），**固定设计对**（录波口径）：
    设计对：B = 10 MHz、T_p = 25.6 μs、L = 256、t_0 = 2 μs、T_L = 1.5 μs、T_s = 6.4 μs
    约束：  b_min = b_max·10⁻⁴（−80 dB）
    扫实际干扰参数：z = 20lg( |h^H y_j(θ*)| / |h^H x| )，误差只进实际干扰

矩阵链（零延迟原位，未变）：J(θ) = diag(j(θ))、H_j = H·diag(j(θ))、X_j 由 x⊙j 铺成。

横轴用**物理量误差 μs**（t_0 / T_L / T_s）；占空比 d = T_L/T_s 是派生量，
在 T_s 固定的图上等价于 T_L（Δd = ΔT_L/T_s）。

网格：三个误差各 ±0.5 μs（论文 Fig.12 的误差区间），步长 0.05 μs（= 半个采样点）。

设计对若不存在会自动跑一次联合设计（约 5 s，N=256）并存到 `结果/设计对/`。

用法：
    .venv/Scripts/python.exe 参数搜索/scripts/run_joint_error_maps.py
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

import baseline  # noqa: E402
from isrj_joint_design.joint_design import joint_design_algorithm  # noqa: E402
from isrj_joint_design.matrices import cross_correlation  # noqa: E402
from isrj_joint_design.params import generate_isrj_sampling_function  # noqa: E402

GRID = np.arange(-0.5, 0.5001, 0.05)          # 三个误差共用的网格 [μs]
DESIGN_NPZ = ROOT / "参数搜索" / "结果" / "设计对" / "joint_design_v2_t0_2us.npz"
OUTDIR = ROOT / "参数搜索" / "结果" / "联合误差"


def design_or_load() -> tuple:
    if DESIGN_NPZ.exists():
        d = np.load(DESIGN_NPZ)
        print(f"  加载设计对：{DESIGN_NPZ.name}")
        return d["x"], d["h"]
    print("  设计对不存在，跑一次联合设计 ...")
    DESIGN_NPZ.parent.mkdir(parents=True, exist_ok=True)
    res = joint_design_algorithm(baseline.params_for(), verbose=False)
    np.savez_compressed(DESIGN_NPZ, x=res["x"], h=res["h"], j=res["j"])
    print(f"    完成（{res['elapsed_s']:.1f} s，PSL {res['psl_db']:.2f} dB）")
    return res["x"], res["h"]


def make_level(x, h):
    peak = abs(np.vdot(h, x))

    def level(e_t0=0.0, e_tl=0.0, e_ts=0.0) -> float:
        p = baseline.params_for(t0_us=baseline.T0_US + e_t0,
                                tau_us=baseline.TAU_US + e_tl,
                                ts_us=baseline.TS_US + e_ts)
        r = cross_correlation(x * generate_isrj_sampling_function(p), h)
        return float(20 * np.log10(abs(r[baseline.N - 1]) / peak + 1e-30))

    return level


def summarize(Z, X, Y, name, xl, yl):
    """沿两个轴的端点分析：两维是否各有内部极小（闭合谷 / 仅一维 / 全单调）。"""
    i0, j0 = int(np.argmin(np.abs(X))), int(np.argmin(np.abs(Y)))
    mx, px = Z[0, j0] - Z[i0, j0], Z[-1, j0] - Z[i0, j0]
    my, py = Z[i0, 0] - Z[i0, j0], Z[i0, -1] - Z[i0, j0]
    xi, yi = mx > 0 and px > 0, my > 0 and py > 0
    shape = ("闭合谷（两维都有内部极小）" if xi and yi
             else (f"仅 {xl} 维有内部极小" if xi else
                   (f"仅 {yl} 维有内部极小" if yi else "两维皆单调")))
    g = np.unravel_index(Z.argmin(), Z.shape)
    return {"map": name, "x_label": xl, "y_label": yl,
            "design_db": float(Z[i0, j0]), "min_db": float(Z.min()),
            "min_at": [float(X[g[0]]), float(Y[g[1]])], "shape": shape,
            "x_minus": mx, "x_plus": px, "y_minus": my, "y_plus": py}


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    OUTDIR.mkdir(parents=True, exist_ok=True)
    print("=== 三对参数联合误差图（v2 口径）===")
    print("  " + baseline.summary().replace(chr(10), chr(10) + "  "))
    x, h = design_or_load()
    level = make_level(x, h)

    maps = [
        ("t0_ts", "Δt_0 (μs)", "ΔT_s (μs)",
         lambda a, b: level(e_t0=a, e_ts=b)),
        ("t0_tl", "Δt_0 (μs)", "ΔT_L (μs)",
         lambda a, b: level(e_t0=a, e_tl=b)),
        ("tl_ts", "ΔT_L (μs)", "ΔT_s (μs)",
         lambda a, b: level(e_tl=a, e_ts=b)),
    ]
    out = {}
    for key, xl, yl, f in maps:
        Z = np.array([[f(a, b) for b in GRID] for a in GRID])
        rows = summarize(Z, GRID, GRID, key, xl, yl)
        out[key] = {"x_label": xl, "y_label": yl, "grid": GRID.tolist(),
                    "level_db": Z.tolist(), "summary": rows}
        print(f"  {key:6s} 设计点 {rows['design_db']:7.2f} dB ；最小 {rows['min_db']:7.2f} dB "
              f"@ ({rows['min_at'][0]:+.2f}, {rows['min_at'][1]:+.2f}) ⇒ {rows['shape']}")
        (OUTDIR / f"{key}_联合误差.json").write_text(json.dumps(out[key], indent=2,
                                                            ensure_ascii=False),
                                                 encoding="utf-8")

    lines = ["# 三对参数联合误差图（v2 口径 · 对齐 Zhou 2026）", "",
             f"设计对：`{DESIGN_NPZ.name}`（B = {baseline.B_HZ / 1e6:g} MHz、"
             f"T_p = {baseline.T_P_S * 1e6:g} μs、L = {baseline.N}、"
             f"t_0 = {baseline.T0_US:g} μs、T_L = {baseline.TAU_US:g} μs、"
             f"T_s = {baseline.TS_US:g} μs、约束 {baseline.PEAK_INTERF_DB:g} dB）。",
             f"三个误差各扫 ±0.5 μs（步长 0.05 μs），网格 21×21。", "",
             "| 参数对 | 设计点 (dB) | 全图最小 (dB) | 最小值位置 | 形态 |", "|---|---|---|---|---|"]
    for key, xl, yl, _ in maps:
        r = out[key]["summary"]
        lines.append(f"| {key} | {r['design_db']:.2f} | {r['min_db']:.2f} | "
                     f"({r['min_at'][0]:+.2f}, {r['min_at'][1]:+.2f}) | {r['shape']} |")
    lines += ["", "占空比 d = T_L/T_s 为派生量：在 T_s 固定的图上，Δd = ΔT_L/T_s。"]
    (OUTDIR / "结果表.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("  JSON 与结果表已保存")


if __name__ == "__main__":
    main()
