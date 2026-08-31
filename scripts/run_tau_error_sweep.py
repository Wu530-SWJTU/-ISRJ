"""截取长度估计误差扫描：实际截取长度偏离设计假设时的性能退化曲线。

设计假设截取长度 2.5 μs（窗口宽 50 点、起点 0）；实际长度 = 2.5 + ε μs。
加载已设计 (x, h)，逐点构造 j_actual（窗口宽 round((2.5+ε)·fs) 点、起点 0），
评估零延迟电平/峰值/IL 相对无误差基准 −30 dB 的退化。

矩阵设计：设计阶段 H_j = H·diag(j_design)；实际输出 r_actual = H·diag(j_actual)·x，
零延迟值 = b_min + h^H(x⊙(j_actual−j_design))。ε>0 新增未约束尾部切片，
ε<0 移除尾部切片（移除量同样未约束），两个方向都可能退化。

用法::

    python scripts/run_tau_error_sweep.py                          # ε ∈ [−1, +1] μs，步长 0.25
    python scripts/run_tau_error_sweep.py --min 0 --max 0.25 --step 0.05  # 近零细扫
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
DESIGN_NPZ = ROOT / "参数误差" / "截取长度" / "joint_design_result_random_phase.npz"
OUTDIR = ROOT / "参数误差" / "截取长度"
TAU_DESIGN_US = 2.5        # 设计假设的截取长度 [μs]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--design-npz", type=Path, default=DESIGN_NPZ)
    ap.add_argument("--min", type=float, default=-1.0, help="长度误差下界 [μs]")
    ap.add_argument("--max", type=float, default=1.0, help="长度误差上界 [μs]")
    ap.add_argument("--step", type=float, default=0.25, help="长度误差步长 [μs]")
    ap.add_argument("--outdir", type=Path, default=OUTDIR)
    args = ap.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    d = np.load(args.design_npz)
    x, h, j_design = d["x"], d["h"], d["j"]
    N = x.size
    peak = abs(np.vdot(h, x))

    sweep = {"tau_error_us": [], "zero_delay_db": [], "peak_db": [], "il": []}
    print(f"{'误差(μs)':>8}  {'实际长度(μs)':>12}  {'零延迟(dB)':>10}  {'峰值(dB)':>9}  {'IL':>12}")
    for e in np.arange(args.min, args.max + 1e-12, args.step):
        tau_actual = TAU_DESIGN_US + e
        p = GlobalParams(N=N, tau=tau_actual * 1e-6, tau_offset=0.0)
        j_actual = generate_isrj_sampling_function(p)
        r = cross_correlation(x * j_actual, h)
        with np.errstate(divide="ignore"):
            zero_db = float(20 * np.log10(abs(r[N - 1]) / peak))
            pk_db = float(20 * np.log10(np.abs(r).max() / peak))
        zero_db = zero_db if np.isfinite(zero_db) else -200.0
        pk_db = pk_db if np.isfinite(pk_db) else -200.0
        il = float(np.sum(np.abs(r) ** 2))
        sweep["tau_error_us"].append(float(e))
        sweep["zero_delay_db"].append(zero_db)
        sweep["peak_db"].append(pk_db)
        sweep["il"].append(il)
        print(f"{e:+8.2f}  {tau_actual:12.2f}  {zero_db:+10.2f}  {pk_db:+9.2f}  {il:12.4e}",
              flush=True)

    args.outdir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.min:g}to{args.max:g}"
    if args.min == -1.0 and args.max == 1.0:
        name = "截取长度有估计误差"          # 默认口径（±1 μs）用正式命名
    else:
        name = f"截取长度有估计误差_{tag}"
    plot_param_error_sweep(sweep, str(args.outdir / f"{name}.png"),
                           xlabel="Window-length error (μs)", xkey="tau_error_us")
    out = args.outdir / f"{name}.json"
    out.write_text(json.dumps(sweep, indent=2), encoding="utf-8")
    print(f"  saved: {out}")


if __name__ == "__main__":
    main()
