"""参数误差扫描：窗口起点误差对已设计波形干扰抑制性能的退化曲线。

加载按"假设参数"（窗口起点 0 μs）设计的最优 (x, h)，把实际窗口起点从
``--min`` 扫到 ``--max``，逐点计算实际干扰经滤波器后的零延迟电平、峰值与
IL，输出退化曲线与 JSON。

研究口径（用户 2026-08-30 约定）：只考虑**正误差**——干扰机截取开始时刻不可能
早于脉冲到达（假设截取从 0 开始，实际截取只能更晚）。

用法::

    python scripts/run_param_error_sweep.py                 # 0 ~ +2.5 μs，步长 0.25
    python scripts/run_param_error_sweep.py --min 0 --max 0.25 --step 0.05
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
from isrj_joint_design.plotting import plot_param_error_sweep  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DESIGN_NPZ = ROOT / "参数误差" / "joint_design_result_random_phase.npz"
OUTDIR = ROOT / "参数误差"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--design-npz", type=Path, default=DESIGN_NPZ)
    ap.add_argument("--min", type=float, default=0.0, help="误差下界 [μs]（研究口径：只考虑正误差）")
    ap.add_argument("--max", type=float, default=2.5, help="误差上界 [μs]")
    ap.add_argument("--step", type=float, default=0.25, help="误差步长 [μs]")
    ap.add_argument("--tau", type=float, default=2.5, help="实际截取长度 [μs]")
    ap.add_argument("--outdir", type=Path, default=OUTDIR)
    args = ap.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    d = np.load(args.design_npz)
    x, h, j_design = d["x"], d["h"], d["j"]
    N = x.size
    peak = abs(np.vdot(h, x))

    sweep = {"offset_error_us": [], "zero_delay_db": [], "peak_db": [], "il": []}
    print(f"{'误差(μs)':>8}  {'零延迟(dB)':>10}  {'峰值(dB)':>9}  {'IL':>12}")
    for e in np.arange(args.min, args.max + 1e-12, args.step):
        p = GlobalParams(N=N, tau=args.tau * 1e-6, tau_offset=(0.0 + e) * 1e-6)
        j_actual = generate_isrj_sampling_function(p)
        r = cross_correlation(x * j_actual, h)
        with np.errstate(divide="ignore"):
            zero_db = float(20 * np.log10(abs(r[N - 1]) / peak))
            pk_db = float(20 * np.log10(np.abs(r).max() / peak))
        zero_db = zero_db if np.isfinite(zero_db) else -200.0   # 窗口完全滑出脉冲
        pk_db = pk_db if np.isfinite(pk_db) else -200.0
        il = float(np.sum(np.abs(r) ** 2))
        sweep["offset_error_us"].append(float(e))
        sweep["zero_delay_db"].append(zero_db)
        sweep["peak_db"].append(pk_db)
        sweep["il"].append(il)
        print(f"{e:+8.2f}  {zero_db:+10.2f}  {pk_db:+9.2f}  {il:12.4e}", flush=True)

    args.outdir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.min:g}to{args.max:g}"
    if args.min == 0.0 and args.max == 2.5:
        name = "转发时间有估计误差"          # 默认口径（正误差 0~2.5 μs）用正式命名
    else:
        name = f"转发时间有估计误差_{tag}"
    plot_param_error_sweep(sweep, str(args.outdir / f"{name}.png"))
    out = args.outdir / f"{name}.json"
    out.write_text(json.dumps(sweep, indent=2), encoding="utf-8")
    print(f"  saved: {out}")


if __name__ == "__main__":
    main()
