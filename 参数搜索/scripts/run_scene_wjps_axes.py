# -*- coding: utf-8 -*-
"""单参数展宽：分别对 t_0 与 T_L 做与 T_s 相同的「次序反馈」展宽，画前后对比图。

口径与 `run_scene_wjps.py` 一致（ADR-0005–0012）：
    5 个误差档（±0.2 μs）、逐档电平块、复合约束、b_min=−50 dB、
    γ=0.6、完整 (x,h) 联合设计、次序反馈调权重直到严格 V。

每个轴单独做，不联合。

用法
    .venv/Scripts/python.exe 参数搜索/scripts/run_scene_wjps_axes.py
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

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import baseline  # noqa: E402
from run_scene_wjps import (  # noqa: E402
    BASE, DEL, ORDER, W_INIT, build_mask, is_v, joint_design_wjps, measure,
)
from scene_joint_design import joint_design_scene  # noqa: E402

OUTDIR = ROOT / "参数搜索" / "结果" / "对抗场景"
AXES = ("t_0", "T_L")


def case_params(axis: str, e: int) -> tuple:
    """返回该误差档的 (t_0, T_L, T_s) 采样点数。"""
    t0, tl, ts = BASE["t_0"], BASE["T_L"], BASE["T_s"]
    if axis == "t_0":
        t0 += e
    elif axis == "T_L":
        tl += e
    else:
        ts += e
    return t0, tl, ts


def case_mask_offsets(axis: str, e: int) -> tuple:
    t0, tl, ts = case_params(axis, e)
    R = max(min(BASE["T_s"] // BASE["T_L"] - 1, ts // tl - 1), 1)
    return build_mask(t0, tl, ts), [r * tl for r in range(1, R + 1)]


def make_family(axis: str, nat: dict) -> list:
    s = sum(nat.values())
    out = []
    for e in ORDER:
        j, offs = case_mask_offsets(axis, e)
        out.append((j, offs, nat[e] / s))
    return out


def measure_axis(x: np.ndarray, h: np.ndarray, axis: str) -> dict:
    """沿指定轴 ±5 个采样点测最差假目标电平。"""
    N = len(x)
    pk = abs(np.vdot(h, x))
    levels = []
    for e in range(-5, 6):
        j, offs = case_mask_offsets(axis, e)
        from scene_joint_design import constraint_vectors
        v = constraint_vectors(x, j, offs, [0] + offs)
        levels.append(max(20 * np.log10(abs(np.vdot(h, vv)) / pk) for vv in v[1:]))
    r = np.convolve(x, np.conj(h[::-1]), mode="full")
    psl = 20 * np.log10(np.delete(np.abs(r), N - 1).max() / pk)
    lpg = 20 * np.log10(pk / np.sqrt(N * np.sum(np.abs(h) ** 2)))
    return {"levels_db": [float(v) for v in levels],
            "psl_db": float(psl), "lpg_db": float(lpg)}


def design_axis(params, axis: str, gamma: float, max_rounds: int = 5) -> dict:
    nat = dict(W_INIT)
    hist = []
    for rnd in range(max_rounds):
        fam = make_family(axis, nat)
        j_hat = sum(w * j for j, _, w in fam)
        res = joint_design_wjps(params, fam, j_hat, gamma, verbose=False)
        m = measure_axis(res["x"], res["h"], axis)
        lv = m["levels_db"][3:8]
        hist.append({"round": rnd, "weights": dict(nat),
                     "levels_db": [float(v) for v in lv],
                     "psl_db": m["psl_db"], "is_v": bool(is_v(lv))})
        print(f"  [{axis}] 第 {rnd + 1} 轮  权重 "
              + ":".join(f"{nat[e]:.1f}" for e in DEL)
              + "  |  " + "  ".join(f"{v:6.1f}" for v in lv)
              + ("   V ✓" if is_v(lv) else "   非单调"), flush=True)
        if is_v(lv):
            return {"x": res["x"], "h": res["h"], "weights": nat,
                    "rounds": rnd + 1, "measured": m, "history": hist}
        center = lv[2]
        for e in DEL:
            if e != 0 and lv[e + 2] < center:
                nat[e] = max(nat[e] * 0.6, 0.2)
        nat[0] = min(nat[0] * 1.6, 40.0)
    return {"x": res["x"], "h": res["h"], "weights": nat,
            "rounds": max_rounds, "measured": m, "history": hist}


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="surrogateescape")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gamma", type=float, default=0.6)
    ap.add_argument("--bmin", type=float, default=-50.0)
    ap.add_argument("--m-outer", type=int, default=100)
    args = ap.parse_args()

    params = dataclasses.replace(
        baseline.params_for(peak_interf_db=args.bmin), M=args.m_outer)
    j0, offs0 = case_mask_offsets("t_0", 0)      # 中心档（三个轴共用）

    print("扩V前：现有设计（同一 b_min）")
    t = time.perf_counter()
    base = joint_design_scene(params, j0, offs0, [0] + offs0, verbose=False)
    print(f"  完成 {time.perf_counter() - t:.1f} s\n")

    results = {}
    for axis in AXES:
        t = time.perf_counter()
        res = design_axis(params, axis, args.gamma)
        print(f"  [{axis}] 完成 {time.perf_counter() - t:.1f} s，"
              f"{res['rounds']} 轮\n")
        before = measure_axis(base["x"], base["h"], axis)
        results[axis] = {"before": before, "after": res["measured"],
                         "weights": {str(e): float(res["weights"][e]) for e in DEL},
                         "rounds": res["rounds"], "history": res["history"]}

    # ---- 绘图：两个轴，前后对比 ----
    RED, GRAY, TXT, BAND = "#B81828", "#A6A6A6", "#7F7F7F", "#FBEEEF"
    plt.rcParams["font.family"] = "Microsoft YaHei"
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8), dpi=200, facecolor="white")
    xs = np.arange(-5, 6)
    names = {"t_0": "$t_0$（转发起始时间）", "T_L": "$T_L$（采样时间长度）"}
    for ax, axis in zip(axes, AXES):
        b = results[axis]["before"]; a = results[axis]["after"]
        ax.axvspan(-2, 2, color=BAND, zorder=0)
        ax.plot(xs, b["levels_db"], color=GRAY, ls="--", lw=1.6, marker="o", ms=3.8,
                zorder=3, label=f"扩V前（PSL {b['psl_db']:.1f} dB）")
        ax.plot(xs, a["levels_db"], color=RED, lw=2.3, marker="s", ms=4.2,
                zorder=4, label=f"扩V后（PSL {a['psl_db']:.1f} dB）")
        ax.set_title(f"{names[axis]} 单独展宽", fontsize=11, color="#404040", pad=8)
        ax.set_xlabel("估计误差（采样点；1 点 = 0.1 $\\mu$s）", fontsize=9, color=TXT)
        ax.set_ylabel("最差假目标电平（dB）", fontsize=9, color=TXT)
        ax.set_xlim(-5.4, 5.4); ax.set_ylim(-56, -15); ax.set_xticks(xs)
        ax.grid(axis="y", color="#E8E8E8", lw=0.6)
        ax.legend(frameon=False, loc="lower center", fontsize=8.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color("#BFBFBF"); ax.spines[sp].set_linewidth(0.7)
        ax.tick_params(labelsize=8, colors=TXT)
    fig.suptitle("单参数展宽：t₀ 与 T_L（权重次序反馈，γ=0.6，b_min=−50 dB）",
                 fontsize=12, color="#404040", y=1.02)
    out = OUTDIR / "图" / "扩V_单参数_t0_TL.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("图已保存：", out)

    payload = {"gamma": args.gamma, "b_min_db": args.bmin, "axes": results}
    jf = OUTDIR / "scene08_wjps_axes_t0_TL.json"
    jf.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("数据已保存：", jf.name)


if __name__ == "__main__":
    main()
