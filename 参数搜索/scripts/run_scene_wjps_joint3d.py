# -*- coding: utf-8 -*-
"""三参数联合展宽（全盒 125 档）：三个轴各有一个模板，联合权重取乘积。

权重（2026-10-25 用户要求「三个参数统一用同一条规则」）：
    ω(δt0, δL, δs) = w_t0(δt0) · w_TL(δL) · w_Ts(δs) / Σ
    每个轴仍是一条对称模板 (1, a, b, a, 1)，只是数值按灵敏度标定。
    **一次设计只有一组 125 个权重**，不存在"分轴各设计一次"。

次序反馈（沿用 ADR-0012）：
    每轮设计后分别检查三条切片；哪条不满足 V，就只调该轴的模板：
        中心不是最低      → 该轴 w₀ ×1.6（上限 60）
        某档 e 低于中心   → 该轴 w_e ×0.6（下限 0.2）
    直到三条切片都成严格 V。

其余口径同 `run_scene_wjps.py`（ADR-0005–0012）：逐档模平方电平块、复合约束、
b_min=−50 dB、γ=0.6、完整 (x,h) 联合设计。

用法
    .venv/Scripts/python.exe 参数搜索/scripts/run_scene_wjps_joint3d.py [--m-outer 100]
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
from run_scene_wjps import BASE, DEL, joint_design_wjps  # noqa: E402
from scene_joint_design import constraint_vectors, joint_design_scene  # noqa: E402

OUTDIR = ROOT / "参数搜索" / "结果" / "对抗场景"
AXES = ("t_0", "T_L", "T_s")
R_MAX = BASE["T_s"] // BASE["T_L"] - 1

# 初始模板（来自单轴反馈的标定）：t_0 6 / T_L 24 / T_s 12
T_INIT = {
    "t_0": {-2: 1.0, -1: 3.0, 0: 6.0, 1: 3.0, 2: 1.0},
    "T_L": {-2: 1.0, -1: 3.0, 0: 24.0, 1: 3.0, 2: 1.0},
    "T_s": {-2: 1.0, -1: 3.0, 0: 12.0, 1: 3.0, 2: 1.0},
}


def build_mask(n_off: int, n_tau: int, n_ts: int, N: int = 256) -> np.ndarray:
    j = np.zeros(N)
    for s in range(4):
        st = n_off + s * n_ts
        if st >= N:
            break
        j[st:min(st + n_tau, N)] = 1.0
    return j


def case_mask_offsets(t0: int, tl: int, ts: int) -> tuple:
    R = max(min(R_MAX, ts // tl - 1), 1)
    return build_mask(t0, tl, ts), [r * tl for r in range(1, R + 1)]


def joint_family(templates: dict) -> list:
    """全盒 125 档，权重 = 三个轴模板的乘积。"""
    s = 1.0
    for ax in AXES:
        s *= sum(templates[ax].values())
    fam = []
    for a in DEL:
        for b in DEL:
            for c in DEL:
                t0, tl, ts = BASE["t_0"] + a, BASE["T_L"] + b, BASE["T_s"] + c
                j, offs = case_mask_offsets(t0, tl, ts)
                w = (templates["t_0"][a] * templates["T_L"][b]
                     * templates["T_s"][c] / s)
                fam.append((j, offs, w))
    c0, o0 = case_mask_offsets(BASE["t_0"], BASE["T_L"], BASE["T_s"])
    fam.sort(key=lambda x: 0 if np.array_equal(x[0], c0) and x[1] == o0 else 1)
    return fam


def slice_levels(x: np.ndarray, h: np.ndarray, axis: str) -> dict:
    N = len(x)
    pk = abs(np.vdot(h, x))
    levels = []
    for e in range(-5, 6):
        t0, tl, ts = BASE["t_0"], BASE["T_L"], BASE["T_s"]
        if axis == "t_0":
            t0 += e
        elif axis == "T_L":
            tl += e
        else:
            ts += e
        j, offs = case_mask_offsets(t0, tl, ts)
        v = constraint_vectors(x, j, offs, [0] + offs)
        levels.append(max(20 * np.log10(abs(np.vdot(h, vv)) / pk) for vv in v[1:]))
    r = np.convolve(x, np.conj(h[::-1]), mode="full")
    psl = 20 * np.log10(np.delete(np.abs(r), N - 1).max() / pk)
    lpg = 20 * np.log10(pk / np.sqrt(N * np.sum(np.abs(h) ** 2)))
    return {"levels_db": [float(v) for v in levels],
            "psl_db": float(psl), "lpg_db": float(lpg)}


def is_v(lv: list) -> bool:
    return (lv[0] > lv[1] > lv[2]) and (lv[2] < lv[3] < lv[4])


def design_joint(params, gamma: float, max_rounds: int = 6) -> dict:
    templates = {ax: dict(T_INIT[ax]) for ax in AXES}
    hist = []
    for rnd in range(max_rounds):
        fam = joint_family(templates)
        j_hat = sum(w * j for j, _, w in fam)
        res = joint_design_wjps(params, fam, j_hat, gamma, verbose=False)
        slices = {ax: slice_levels(res["x"], res["h"], ax) for ax in AXES}
        all_v = all(is_v(slices[ax]["levels_db"][3:8]) for ax in AXES)
        hist.append({"round": rnd,
                     "templates": {ax: dict(templates[ax]) for ax in AXES},
                     "slices": {ax: slices[ax]["levels_db"][3:8] for ax in AXES},
                     "is_v": bool(all_v)})
        print(f"  第 {rnd + 1} 轮", flush=True)
        for ax in AXES:
            print(f"    模板 {ax:>4} "
                  + ":".join(f"{templates[ax][e]:.1f}" for e in DEL), flush=True)
            lv = slices[ax]["levels_db"][3:8]
            print(f"      {ax:>4}: " + "  ".join(f"{v:6.1f}" for v in lv)
                  + ("   V ✓" if is_v(lv) else "   非单调"), flush=True)
        if all_v:
            return {"x": res["x"], "h": res["h"], "templates": templates,
                    "rounds": rnd + 1, "slices": slices, "history": hist}
        for ax in AXES:
            lv = slices[ax]["levels_db"][3:8]
            c = lv[2]
            if c != min(lv):
                templates[ax][0] = min(templates[ax][0] * 1.6, 60.0)
            for e in DEL:
                if e != 0 and lv[e + 2] < c:
                    templates[ax][e] = max(templates[ax][e] * 0.6, 0.2)
            # 相邻次序反转：外层低于内层 → 减外层的权（让它抬起来）
            for inner in (-1, 1):
                outer = 2 * inner
                if lv[outer + 2] < lv[inner + 2]:
                    templates[ax][outer] = max(templates[ax][outer] * 0.6, 0.2)
    return {"x": res["x"], "h": res["h"], "templates": templates,
            "rounds": max_rounds, "slices": slices, "history": hist}


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gamma", type=float, default=0.6)
    ap.add_argument("--bmin", type=float, default=-50.0)
    ap.add_argument("--m-outer", type=int, default=100)
    ap.add_argument("--max-rounds", type=int, default=6)
    args = ap.parse_args()

    params = dataclasses.replace(
        baseline.params_for(peak_interf_db=args.bmin), M=args.m_outer)
    j0, offs0 = case_mask_offsets(BASE["t_0"], BASE["T_L"], BASE["T_s"])

    print("扩V前：现有设计")
    t = time.perf_counter()
    base = joint_design_scene(params, j0, offs0, [0] + offs0, verbose=False)
    print(f"  完成 {time.perf_counter() - t:.1f} s")

    print(f"扩V后：三参数联合（125 档），γ={args.gamma:g}，M={params.M}，"
          f"三模板分别反馈")
    t = time.perf_counter()
    res = design_joint(params, args.gamma, args.max_rounds)
    print(f"  完成 {time.perf_counter() - t:.1f} s，{res['rounds']} 轮\n")

    out = {"gamma": args.gamma, "b_min_db": args.bmin, "m_outer": params.M,
           "family": 125,
           "weights": "per-axis templates (product), one 125-vector per design",
           "templates_final": {ax: {str(e): float(res["templates"][ax][e])
                                    for e in DEL} for ax in AXES},
           "rounds": res["rounds"], "history": res["history"], "slices": {}}
    xs = list(range(-5, 6))
    print(f"{'轴':>5}{'状态':>6}" + "".join(f"{e:+7d}" for e in xs))
    for ax in AXES:
        before = slice_levels(base["x"], base["h"], ax)
        after = res["slices"][ax]
        out["slices"][ax] = {"before": before, "after": after}
        for tag, r in (("前", before), ("后", after)):
            print(f"{ax:>5}{tag:>6}" + "".join(f"{v:7.1f}" for v in r["levels_db"])
                  + f"  | PSL {r['psl_db']:6.2f}")
    (OUTDIR / "scene08_wjps_joint3d.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n数据已保存：scene08_wjps_joint3d.json")


if __name__ == "__main__":
    main()
