"""评估侧实验：两次转发的 ISRJ 干扰经设计滤波器后的压制效果。

设计阶段仍按单份零延迟干扰模型（论文/PRD 结构，模块二~五不动）；设计完成后
生成多转发干扰信号并过 h 做脉压，观察压制水平。

实验设定（用户 2026-08-30）：
    - 采样窗口 [0, 2.5) μs，采样周期 T_s = 20 μs（tau_offset=0, T_s=20e-6）
    - 转发#1 在截取完成后马上转发（偏移 tau = 2.5 μs = 50 点）
    - 转发#2 在采样开始 10 μs 后（偏移 200 点）

用法::

    python scripts/run_multi_repeat_eval.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from isrj_joint_design.joint_design import joint_design_algorithm, summarize  # noqa: E402
from isrj_joint_design.matrices import cross_correlation  # noqa: E402
from isrj_joint_design.params import (  # noqa: E402
    GlobalParams,
    generate_isrj_jamming_signal,
    generate_isrj_sampling_function,
)
from isrj_joint_design.plotting import plot_multi_repeat_jamming  # noqa: E402

FIGURES2 = Path(__file__).resolve().parents[1] / "figures_转发2次"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--M", type=int, default=100)
    ap.add_argument("--seed", type=int, default=2020)
    ap.add_argument("--outdir", type=Path, default=FIGURES2)
    args = ap.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    # 本实验自己的口径显式钉住（不依赖 GlobalParams 默认值，默认值恢复 PRD 基线后
    # 本脚本口径不受影响）：转发#1 截取完成后发（偏移 2.5 μs）、转发#2 在 +10 μs。
    params = GlobalParams(N=800, T_s=20e-6, tau_offset=0.0, M=args.M, seed=args.seed,
                          retransmit_offsets=(2.5e-6, 10e-6))
    j = generate_isrj_sampling_function(params)
    print(f"采样函数: 周期 {params.N_Ts} 点, 窗口 {params.N_tau} 点, "
          f"非零 {int(j.sum())} 点（每 20 μs 采样一次）", flush=True)
    print(f"转发偏移: {[f'{d*1e6:.0f} μs' for d in params.retransmit_offsets]}\n", flush=True)

    print("=== 设计阶段（单份零延迟干扰模型，模块二~五不变） ===", flush=True)
    result = joint_design_algorithm(params, j)
    x, h = result["x"], result["h"]
    print("\n" + summarize(result, params) + "\n", flush=True)

    print("=== 评估阶段：双转发干扰经滤波器的脉压输出 ===", flush=True)
    y_jam = generate_isrj_jamming_signal(x, params)
    xj = x * j
    r_single = cross_correlation(xj, h)
    r_multi = cross_correlation(y_jam, h)
    peak = abs(np.vdot(h, x))

    report = {
        "single_copy_il": float(np.sum(np.abs(r_single) ** 2)),
        "multi_copy_il": float(np.sum(np.abs(r_multi) ** 2)),
        "multi_copy_peak_db": float(20 * np.log10(np.abs(r_multi).max() / peak)),
        "multi_copy_zero_delay_db": float(
            20 * np.log10(abs(r_multi[params.N - 1]) / peak)),
        "copy_positions_db": [
            float(20 * np.log10(abs(r_multi[params.N - 1 + int(round(d * params.fs))]) / peak))
            for d in params.retransmit_offsets
        ],
    }
    print(f"  单份转发输出能量 IL      : {report['single_copy_il']:.4e}")
    print(f"  双份转发输出能量 IL      : {report['multi_copy_il']:.4e}  "
          f"(比值 {report['multi_copy_il']/report['single_copy_il']:.3f})")
    print(f"  双转发输出整体峰值       : {report['multi_copy_peak_db']:.2f} dB")
    print(f"  双转发在零延迟处的电平   : {report['multi_copy_zero_delay_db']:.2f} dB")
    for i, d in enumerate(params.retransmit_offsets, 1):
        print(f"  转发#{i} 主瓣位置电平     : {report['copy_positions_db'][i-1]:.2f} dB"
              f"  (偏移 {d*1e6:g} μs)")

    args.outdir.mkdir(parents=True, exist_ok=True)
    # 文件名带偏移标记，不同偏移口径的实验互不覆盖
    tag = "offs" + "_".join(f"{d * 1e6:g}" for d in params.retransmit_offsets)
    print("\n绘图:")
    plot_multi_repeat_jamming(x, h, params,
                              str(args.outdir / f"fig_multi_repeat_jamming_{tag}.png"))
    out = args.outdir / f"multi_repeat_eval_{tag}.json"
    out.write_text(json.dumps({**report, "offsets_us": [d * 1e6 for d in params.retransmit_offsets]},
                              indent=2), encoding="utf-8")
    print(f"  saved: {out}")


if __name__ == "__main__":
    main()
