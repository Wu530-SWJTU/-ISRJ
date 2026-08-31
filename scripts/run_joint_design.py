"""复现论文图 1 与图 2：跑一次完整联合设计并输出图像。

用法::

    python scripts/run_joint_design.py                  # 论文设定 N=800, M=100
    python scripts/run_joint_design.py --M 20 --N 256   # 快速试跑
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from isrj_joint_design.joint_design import joint_design_algorithm, summarize  # noqa: E402
from isrj_joint_design.params import GlobalParams, generate_isrj_sampling_function  # noqa: E402
from isrj_joint_design.plotting import (  # noqa: E402
    plot_convergence,
    plot_correlation_comparison,
    plot_waveform,
)

FIGURES = Path(__file__).resolve().parents[1] / "figures_初始0"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--N", type=int, default=800, help="码长（默认论文值 800）")
    ap.add_argument("--M", type=int, default=100, help="外层迭代次数（默认论文值 100）")
    ap.add_argument("--eta", type=float, default=1.1, help="PAPR 约束")
    ap.add_argument("--rho", type=float, default=5000.0, help="ADMM 惩罚参数")
    ap.add_argument("--tau", type=float, default=2.5e-6,
                    help="中断采样脉宽 [s]（PRD 默认 2.5e-6）")
    ap.add_argument("--tau-offset", type=float, default=0.0,
                    help="采样窗口起点 [s]（PRD 默认 0，即窗口 [0, 2.5) μs；"
                    "实验旋钮，如 2.5e-6 表示窗口 [2.5, 5) μs）")
    ap.add_argument("--seed", type=int, default=2020)
    ap.add_argument("--init", choices=("random_phase", "lfm"),
                    default="random_phase", help="波形初始化方式（PRD_01 默认随机相位；"
                    "LFM 初值更接近论文 -0.9dB 的处理增益损失，见 CLAUDE.md）")
    ap.add_argument("--outdir", type=Path, default=FIGURES)
    args = ap.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    params = GlobalParams(N=args.N, T_p=args.N / 20e6, M=args.M,
                          eta=args.eta, rho=args.rho, seed=args.seed, init=args.init,
                          tau=args.tau, tau_offset=args.tau_offset)
    j = generate_isrj_sampling_function(params)

    print(f"联合设计: N={params.N}, M={params.M}, eta={params.eta}, "
          f"rho={params.rho}, epsilon={params.epsilon}", flush=True)
    result = joint_design_algorithm(params, j)

    print()
    print(summarize(result, params))

    args.outdir.mkdir(parents=True, exist_ok=True)
    # 文件名带 init 标记，不同初始化方式的结果互不覆盖
    tag = args.init
    print("\n绘图:")
    plot_convergence(result["isl_history"], result["il_history"],
                     str(args.outdir / f"fig1a_convergence_{tag}.png"))
    plot_waveform(result["x"], params, str(args.outdir / f"fig1b_waveform_{tag}.png"))
    plot_correlation_comparison(result["x"], result["h"], j, params,
                                str(args.outdir / f"fig2_correlation_{tag}.png"))

    npz = args.outdir / f"joint_design_result_{tag}.npz"
    np.savez(npz, x=result["x"], h=result["h"], j=j,
             isl_history=result["isl_history"], il_history=result["il_history"])
    print(f"  saved: {npz}")


if __name__ == "__main__":
    main()
