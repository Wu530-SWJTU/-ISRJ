"""评估侧实验：干扰参数估计误差对已设计波形的影响。

背景：联合设计默认干扰参数完全已知（采样窗口起点、截取长度、转发方式）。实际中
这些参数是**估计**出来的，存在误差。本脚本加载按"假设参数"设计好的最优 (x, h)，
构造带误差的"实际干扰"信号，评估其经滤波器后的输出相对 −30 dB 基准的退化。

矩阵设计：设计阶段 H_j = H·diag(j_design)、约束 h^H(x⊙j_design) = b_min；
实际干扰 x⊙j_actual 的输出 r_actual = H·diag(j_actual)·x（同一 H、只换 diag）。
窗口起点误差 δ 使实际零延迟值 = b_min + h^H(x⊙(j_actual−j_design))，第二项未约束。

用法::

    python scripts/run_param_error_eval.py                     # 窗口起点误差 +1 μs
    python scripts/run_param_error_eval.py --offset-error 2    # 误差 +2 μs
    python scripts/run_param_error_eval.py --offset-error -1   # 误差 −1 μs
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from isrj_joint_design.matrices import cross_correlation  # noqa: E402
from isrj_joint_design.params import GlobalParams, generate_isrj_sampling_function  # noqa: E402
from isrj_joint_design.plotting import plot_param_error_result  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DESIGN_NPZ = ROOT / "参数误差" / "joint_design_result_random_phase.npz"
OUTDIR = ROOT / "参数误差"


def _db(values: np.ndarray, ref: float) -> np.ndarray:
    return 20 * np.log10(np.abs(values) / ref)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--design-npz", type=Path, default=DESIGN_NPZ,
                    help="已设计 (x, h, j) 的 npz 记录")
    ap.add_argument("--offset-error", type=float, default=1.0,
                    help="实际窗口起点相对设计假设的误差 [μs]")
    ap.add_argument("--tau", type=float, default=2.5,
                    help="实际截取长度 [μs]（默认与设计假设一致）")
    ap.add_argument("--outdir", type=Path, default=OUTDIR)
    args = ap.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    d = np.load(args.design_npz)
    x, h, j_design = d["x"], d["h"], d["j"]
    N = x.size
    peak = abs(np.vdot(h, x))

    p_actual = GlobalParams(N=N, tau=args.tau * 1e-6,
                            tau_offset=(0.0 + args.offset_error) * 1e-6)
    j_actual = generate_isrj_sampling_function(p_actual)

    r_assumed = cross_correlation(x * j_design, h)     # 假设参数下的输出（设计基准）
    r_actual = cross_correlation(x * j_actual, h)      # 实际参数下的输出

    report = {
        "offset_error_us": args.offset_error,
        "tau_us": args.tau,
        "assumed_zero_delay_db": float(_db(r_assumed[N - 1], peak)),
        "actual_zero_delay_db": float(_db(r_actual[N - 1], peak)),
        "assumed_peak_db": float(_db(np.max(np.abs(r_assumed)), peak)),
        "actual_peak_db": float(_db(np.max(np.abs(r_actual)), peak)),
        "il_assumed": float(np.sum(np.abs(r_assumed) ** 2)),
        "il_actual": float(np.sum(np.abs(r_actual) ** 2)),
    }

    print("=== 干扰参数误差评估 ===")
    print(f"  设计假设: 窗口起点 0 μs、截取长度 2.5 μs（零延迟输出钉 -30 dB）")
    print(f"  实际干扰: 窗口起点 {args.offset_error:+g} μs（误差 "
          f"{args.offset_error:+g} μs = {round(args.offset_error*20):+d} 点）、"
          f"截取长度 {args.tau:g} μs")
    print(f"  假设参数下: 零延迟 {report['assumed_zero_delay_db']:.2f} dB、"
          f"峰值 {report['assumed_peak_db']:.2f} dB、IL {report['il_assumed']:.4e}")
    print(f"  实际参数下: 零延迟 {report['actual_zero_delay_db']:.2f} dB、"
          f"峰值 {report['actual_peak_db']:.2f} dB、IL {report['il_actual']:.4e}")
    print(f"  零延迟退化: {report['actual_zero_delay_db'] - report['assumed_zero_delay_db']:+.2f} dB")

    args.outdir.mkdir(parents=True, exist_ok=True)
    tag = f"offs_err{args.offset_error:g}us"
    plot_param_error_result(x, h, j_actual, peak, args.offset_error,
                            str(args.outdir / f"fig_param_error_{tag}.png"))
    out = args.outdir / f"param_error_eval_{tag}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"  saved: {out}")


if __name__ == "__main__":
    main()
