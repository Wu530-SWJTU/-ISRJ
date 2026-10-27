# -*- coding: utf-8 -*-
"""精确命中实验：扩V前 / 扩V后，BO 能否在预算内锁定 θ*。

口径（2026-10-25 用户定）
    判据   量化后的 θ̂ 与 θ* 完全相同。栅格 1 点 = 0.1 μs，0 < 容差 < 0.1 只含
           误差为 0 的点，所以 0.02 μs 与"精确命中"等价。
    停止   首次评估落在 θ* 即停；预算耗尽算失败。
    记录   全量 trace（θ̂、f、单次耗时、来源），判据可事后按 ±1 采样点重算。
    条件   before：单点联合设计；after：125 档三维展宽族（templates 取自
           scene08_wjps_joint3d.json）。两者 b_min 均为 −50 dB、γ=0.6、M=100。
    搜索   三维整数栅格，各轴 ±5 采样点（1331 格），初始失配 (+5,+5,+5)；
           GP-EI 超参与 run_scene_bo_search.py 一致。

用法
    .venv/Scripts/python.exe 参数搜索/scripts/run_scene_bo_precision.py \
        --condition before --budget 200 --seed 0
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import baseline  # noqa: E402
from scipy.stats import norm  # noqa: E402

from run_bo_search import gp_predict  # noqa: E402
from run_scene_bo_search import (  # noqa: E402
    ELL, FTS_KM, HI, J_TRUE, KEYS, LO, LOCAL_SIGMA, N_CAND, N_INIT_RAND,
    N_LOCAL, OFFS_TRUE, TAU_T, THETA_INIT, THETA_TRUE,
)
from run_scene_error_sweep_3param import (  # noqa: E402
    build_mask, jamming_profile_scene,
)
from run_scene_wjps import joint_design_wjps  # noqa: E402
import run_scene_wjps_joint3d as J  # noqa: E402
from scene_joint_design import joint_design_scene  # noqa: E402

OUTDIR = ROOT / "参数搜索" / "结果" / "对抗场景"
AXES = ("t_0", "T_L", "T_s")


def to_theta(u: np.ndarray) -> dict:
    v = np.round(LO + np.clip(u, 0, 1) * (HI - LO))
    return {k: int(v[i]) for i, k in enumerate(KEYS)}


def load_templates() -> dict:
    d = json.loads((OUTDIR / "scene08_wjps_joint3d.json").read_text(encoding="utf-8"))
    return {ax: {int(k): float(v) for k, v in d["templates_final"][ax].items()}
            for ax in AXES}


def family_at(center: dict, templates: dict) -> list:
    """把 125 档族平移到 center（权重只依赖相对档位）。"""
    s = 1.0
    for ax in AXES:
        s *= sum(templates[ax].values())
    fam = []
    for a in J.DEL:
        for b in J.DEL:
            for c in J.DEL:
                j, offs = J.case_mask_offsets(center["t_0"] + a,
                                              center["T_L"] + b,
                                              center["T_s"] + c)
                w = (templates["t_0"][a] * templates["T_L"][b]
                     * templates["T_s"][c] / s)
                fam.append((j, offs, w))
    return fam


def design_at(theta_hat: dict, condition: str, params, templates) -> tuple:
    if condition == "before":
        j = build_mask(theta_hat["t_0"], theta_hat["T_L"], theta_hat["T_s"])
        R = max(min(J.R_MAX, theta_hat["T_s"] // theta_hat["T_L"] - 1), 1)
        offs = [r * theta_hat["T_L"] for r in range(1, R + 1)]
        res = joint_design_scene(params, j, offs, [0] + offs, verbose=False)
    else:
        fam = family_at(theta_hat, templates)
        j_hat = sum(w * j for j, _, w in fam)
        res = joint_design_wjps(params, fam, j_hat, 0.6, verbose=False)
    return res["x"], res["h"]


def max_err_samples(theta: dict) -> int:
    return max(abs(theta[k] - THETA_TRUE[k]) for k in KEYS)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", choices=("before", "after"), required=True)
    ap.add_argument("--budget", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--m-outer", type=int, default=100)
    args = ap.parse_args()

    params = dataclasses.replace(baseline.params_for(peak_interf_db=-50.0),
                                 M=args.m_outer)
    templates = load_templates() if args.condition == "after" else None

    print(f"=== 精确命中实验：{args.condition}（b_min=−50 dB，种子 {args.seed}）===")
    print(f"  θ* = {THETA_TRUE}；θ̂₀ = {THETA_INIT}；预算 {args.budget} 次")
    print("  判据：量化后 θ̂ == θ*（0.02 μs 与精确命中等价）\n")

    rng = np.random.default_rng(args.seed)
    pts, vals, trace = [], [], []

    def evaluate(u: np.ndarray, source: str) -> None:
        th = to_theta(u)
        t0 = time.perf_counter()
        x, h = design_at(th, args.condition, params, templates)
        r_km, y_jam = jamming_profile_scene(x, h, J_TRUE, OFFS_TRUE, TAU_T)
        f = max(float(y_jam[int(np.argmin(np.abs(r_km - p)))]) for p in FTS_KM)
        dt = time.perf_counter() - t0
        pts.append(np.asarray(u, float))
        vals.append(f)
        trace.append({"i": len(trace) + 1, "theta": th, "f_db": f,
                      "err_samples": max_err_samples(th), "source": source,
                      "dt_s": dt})
        print(f"  [{len(trace):3d}] θ̂=({th['t_0']:2d},{th['T_L']:2d},{th['T_s']:2d})"
              f"  err={max_err_samples(th)}  f={f:7.2f} dB  ({dt:5.1f} s)  {source}",
              flush=True)

    def hit() -> bool:
        return trace[-1]["err_samples"] == 0

    t_start = time.perf_counter()
    u_init = (np.array([THETA_INIT[k] for k in KEYS], float) - LO) / (HI - LO)
    evaluate(u_init, "init")
    while len(trace) < 1 + N_INIT_RAND and not hit():
        evaluate(rng.uniform(0, 1, 3), "init")

    while len(trace) < args.budget and not hit():
        X, y = np.array(pts), np.array(vals)
        fmin = float(y.min())
        s2 = max(float(y.var()), 1e-3)
        n_lat = HI - LO
        cand = np.round(rng.uniform(0, 1, (N_CAND, 3)) * n_lat) / n_lat
        ib = int(y.argmin())
        loc = np.round((X[ib] + rng.normal(0, LOCAL_SIGMA, (N_LOCAL, 3)))
                       * (HI - LO)) / (HI - LO)
        cand = np.clip(np.vstack([cand, loc]), 0, 1)
        seen = {tuple(np.round(pt * n_lat).astype(int)) for pt in X}
        keep = np.array([tuple(np.round(c * n_lat).astype(int)) not in seen
                         for c in cand])
        cand = cand[keep]
        m, var = gp_predict(X, y, cand, s2, ELL)
        s = np.sqrt(var)
        zz = (fmin - m) / s
        ei = (fmin - m) * norm.cdf(zz) + s * norm.pdf(zz)
        evaluate(cand[int(ei.argmax())], "bo")

    elapsed = time.perf_counter() - t_start
    exact_idx = next((t["i"] for t in trace if t["err_samples"] == 0), None)
    near_idx = next((t["i"] for t in trace if t["err_samples"] <= 1), None)
    best = min(trace, key=lambda t: t["f_db"])

    print("\n=== 结果 ===")
    print(f"  耗时          : {elapsed:.1f} s（{elapsed / 60:.1f} min）")
    print(f"  评估次数      : {len(trace)} / {args.budget}")
    print(f"  精确命中      : {'第 %d 次' % exact_idx if exact_idx else '未命中'}")
    if near_idx:
        src = trace[near_idx - 1]["source"]
        print(f"  首次 ±1 命中  : 第 {near_idx} 次（{src}）")
    print(f"  最小误差      : {min(t['err_samples'] for t in trace)} 采样点")
    print(f"  最好电平      : {best['f_db']:.2f} dB @ {best['theta']}")

    out = OUTDIR / (f"scene08_bo_precision_{args.condition}"
                    f"_seed{args.seed}_budget{args.budget}.json")
    out.write_text(json.dumps({
        "condition": args.condition, "b_min_db": -50.0, "gamma": 0.6,
        "m_outer": args.m_outer, "seed": args.seed, "budget": args.budget,
        "theta_true": THETA_TRUE, "theta_init": THETA_INIT,
        "elapsed_s": elapsed, "n_evals": len(trace),
        "exact_hit_eval": exact_idx, "near_hit_eval": near_idx,
        "best_f_db": best["f_db"], "best_theta": best["theta"],
        "trace": trace,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  已保存        : {out.name}")


if __name__ == "__main__":
    main()
