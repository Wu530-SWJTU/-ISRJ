"""复现论文图 3：对 PAPR ∈ {1, 1.05, 1.1, 1.15, 1.21, 1.44} 分别做联合设计。

用法::

    python scripts/run_papr_sweep.py                # 论文设定 N=800, M=100
    python scripts/run_papr_sweep.py --M 20 --N 256 # 快速试跑
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from isrj_joint_design.joint_design import joint_design_algorithm  # noqa: E402
from isrj_joint_design.params import (  # noqa: E402
    PAPR_RANGE,
    GlobalParams,
    generate_isrj_sampling_function,
)
from isrj_joint_design.plotting import plot_papr_sweep  # noqa: E402

FIGURES = Path(__file__).resolve().parents[1] / "figures"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--N", type=int, default=800)
    ap.add_argument("--M", type=int, default=100)
    ap.add_argument("--rho", type=float, default=5000.0)
    ap.add_argument("--seed", type=int, default=2020)
    ap.add_argument("--papr", type=float, nargs="*", default=list(PAPR_RANGE))
    ap.add_argument("--init", choices=("random_phase", "lfm"),
                    default="random_phase", help="波形初始化方式。PRD 默认随机相位；"
                    "复现论文图 3(a) 的处理增益损失单调趋势请用 --init lfm"
                    "（见 CLAUDE.md 已知偏差）")
    ap.add_argument("--repeats", type=int, default=1,
                    help="每个 PAPR 的多起点次数，取目标函数最优的一次。问题非凸，"
                         "单起点的随机波动（约 0.3dB）会淹没 PAPR 本身的趋势")
    ap.add_argument("--outdir", type=Path, default=FIGURES)
    args = ap.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    sweep = {"papr": [], "loss_db": [], "isl": [], "il": [], "psl_db": [],
             "isrj_peak_db": [], "achieved_papr": [], "loss_spread_db": [],
             "loss_best_db": []}
    all_runs: dict[str, list] = {}

    for eta in args.papr:
        print(f"\n=== PAPR eta = {eta} ===", flush=True)
        runs = []
        for k in range(args.repeats):
            params = GlobalParams(N=args.N, T_p=args.N / 20e6, M=args.M,
                                  eta=eta, rho=args.rho, seed=args.seed + k,
                                  init=args.init)
            j = generate_isrj_sampling_function(params)
            runs.append(joint_design_algorithm(params, j, log_every=50))
        eps = GlobalParams().epsilon
        # 两种选取策略并存：目标最优（与联合设计一致）与损失最优（对应图 3(a) 的
        # 趋势验收）。两者对非凸单起点波动鲁棒性不同，均写入输出。
        r = min(runs, key=lambda v: eps * v["final_isl"] + (1 - eps) * v["final_il"])
        r_loss = min(runs, key=lambda v: v["processing_gain_loss_db"])
        losses = [v["processing_gain_loss_db"] for v in runs]
        sweep["papr"].append(eta)
        sweep["loss_db"].append(r["processing_gain_loss_db"])
        sweep["isl"].append(r["final_isl"])
        sweep["il"].append(r["final_il"])
        sweep["psl_db"].append(r["psl_db"])
        sweep["isrj_peak_db"].append(r["isrj_peak_db"])
        sweep["achieved_papr"].append(r["papr"])
        sweep["loss_spread_db"].append(max(losses) - min(losses))
        sweep["loss_best_db"].append(r_loss["processing_gain_loss_db"])
        all_runs[str(eta)] = [
            {"seed": args.seed + k,
             "loss_db": v["processing_gain_loss_db"],
             "isl": v["final_isl"], "il": v["final_il"], "psl_db": v["psl_db"]}
            for k, v in enumerate(runs)
        ]
        print(f"  -> [目标最优] loss={r['processing_gain_loss_db']:.3f} dB  "
              f"ISLs={r['final_isl']:.4e}  ILs={r['final_il']:.4e}  "
              f"PAPR={r['papr']:.4f}  (起点间 loss 极差 "
              f"{max(losses) - min(losses):.3f} dB)", flush=True)
        print(f"  -> [损失最优] loss={r_loss['processing_gain_loss_db']:.3f} dB  "
              f"ISLs={r_loss['final_isl']:.4e}  ILs={r_loss['final_il']:.4e}  "
              f"PSL={r_loss['psl_db']:.1f} dB", flush=True)

    args.outdir.mkdir(parents=True, exist_ok=True)
    # 文件名带 init 与 repeats，不同初始化方式的结果互不覆盖
    tag = args.init if args.repeats == 1 else f"{args.init}_r{args.repeats}"
    print("\n绘图:")
    plot_papr_sweep(sweep, str(args.outdir / f"fig3_papr_sweep_{tag}.png"))
    out = args.outdir / f"papr_sweep_{tag}.json"
    out.write_text(json.dumps(sweep, indent=2), encoding="utf-8")
    print(f"  saved: {out}")
    out = args.outdir / f"papr_sweep_all_runs_{tag}.json"
    out.write_text(json.dumps(all_runs, indent=2), encoding="utf-8")
    print(f"  saved: {out}")


if __name__ == "__main__":
    main()
