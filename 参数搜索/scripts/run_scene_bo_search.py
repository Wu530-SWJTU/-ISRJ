# -*- coding: utf-8 -*-
"""对抗场景第 5 步：三参数联合的贝叶斯优化搜索（无 WJPS 展宽口径）。

问题：雷达对干扰参数只有**初始失配的估计** θ̂_init，要靠观测（距离像）反推
真实的 θ*，使抗干扰效果恢复。搜索量 θ̂ = (t_0, T_L, T_s)。

口径（2026-10-01 用户定）：
    目标函数 f(θ̂) = **干扰单独**在三个假目标处的**最差归一化电平**（dB，含 J/S）
                     = max_k 20log10(|干扰响应| / |主瓣响应|)
       —— 不用距离像本身：它被噪声底（−31.2 dB）与旁瓣底顶着，再优化无意义；
          这也与搜索线 C 的口径一致（只量干扰）。
    成功判据 f(θ̂) < **−35 dB**（沉到噪声底/旁瓣底以下、假目标不可见）或预算耗尽。
    搜索空间 = **整数采样点格**（掩膜由整数点决定，格内 f 相同 ⇒ 只搜栅格）。
    每次评估 = 完整闭环：按 θ̂ **重新联合设计 (x,h)**（≈4.7 s）→ 生成干扰 → 量电平。

算法：贝叶斯优化（GP-EI），复用搜索线 `run_bo_search.py` 的 GP/EI 实现。
      GP 核 Matérn 3/2、ℓ=0.15（归一化；= 1.5 个采样点）；每轮 2000 栅格候选
      + 200 个围绕当前最优的高斯扰动（σ=0.08，取整到栅格）。

用法（预计 4~6 分钟）：
    .venv/Scripts/python.exe 参数搜索/scripts/run_scene_bo_search.py
"""

from __future__ import annotations

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
from scipy.stats import norm  # noqa: E402

import baseline  # noqa: E402
from run_bo_search import gp_predict  # noqa: E402  （复用搜索线的 GP 实现）
from run_scene_error_sweep_3param import (  # noqa: E402
    BASE, R_MAX, build_mask, design_for, jamming_profile_scene, scene_profile,
)
from run_scene_jamming import (  # noqa: E402
    KM_PER_US, R_T_KM, TAU_T_US, TEXT_PRIMARY, TEXT_SECONDARY, SURFACE,
)

OUTDIR = ROOT / "参数搜索" / "结果" / "对抗场景"

# ---- 搜索设定 ----
THETA_TRUE = dict(BASE)                                    # 真实干扰机（基线）
HALF_BOX = 5                                               # 每轴 ±5 采样点
THETA_INIT = {k: BASE[k] + HALF_BOX for k in BASE}         # 初始失配（+5,+5,+5）
TARGET_DB = -35.0                                          # 成功判据
MAX_EVALS = 50                                             # 评估预算
N_INIT_RAND = 4                                            # 初始随机点数
ELL = 0.15                                                 # GP 长度尺度（归一化）
N_CAND, N_LOCAL, LOCAL_SIGMA = 2000, 200, 0.08

KEYS = ["t_0", "T_L", "T_s"]
LO = np.array([BASE[k] - HALF_BOX for k in KEYS], dtype=float)
HI = np.array([BASE[k] + HALF_BOX for k in KEYS], dtype=float)
TRUE_U = (np.array([THETA_TRUE[k] for k in KEYS], float) - LO) / (HI - LO)
J_TRUE = build_mask(BASE["t_0"], BASE["T_L"], BASE["T_s"])
OFFS_TRUE = [r * BASE["T_L"] for r in range(1, R_MAX + 1)]
TAU_T = int(round(TAU_T_US / baseline.SAMPLE_US))
FTS_KM = [R_T_KM + o * baseline.SAMPLE_US * KM_PER_US for o in OFFS_TRUE]


def to_theta(u: np.ndarray) -> dict:
    """归一化坐标 → 整数采样点参数。"""
    v = np.round(LO + np.clip(u, 0, 1) * (HI - LO))
    return {k: int(v[i]) for i, k in enumerate(KEYS)}


def objective(u: np.ndarray) -> float:
    """闭环评估：按 θ̂ 重新设计，量干扰单独在三个假目标处的最差归一化电平。"""
    theta_hat = to_theta(u)
    x, h = design_for(theta_hat)
    r_km, y_jam = jamming_profile_scene(x, h, J_TRUE, OFFS_TRUE, TAU_T)
    lv = [float(y_jam[int(np.argmin(np.abs(r_km - p)))]) for p in FTS_KM]
    return max(lv)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    OUTDIR.mkdir(parents=True, exist_ok=True)

    budget = MAX_EVALS
    if "--max-evals" in sys.argv:          # 预算可用命令行覆盖
        budget = int(sys.argv[sys.argv.index("--max-evals") + 1])
    seed = 0
    if "--seed" in sys.argv:               # 随机种子（换种子 = 独立重跑）
        seed = int(sys.argv[sys.argv.index("--seed") + 1])
    suffix = "" if budget == MAX_EVALS else f"_{budget}evals"
    if seed != 0:
        suffix += f"_seed{seed}"

    if "--plot-only" in sys.argv:          # 从已保存的 JSON 重画，不重跑搜索
        d = json.loads((OUTDIR / f"scene07_bo_search{suffix}.json").read_text(
            encoding="utf-8"))
        make_figure(d["theta_init"], d["theta_best"], d, suffix)
        return

    print("=== 三参数联合贝叶斯优化搜索（无 WJPS 展宽）===")
    print(f"  真实干扰机 θ* = {THETA_TRUE}（采样点）；假目标固定 +1.5/+3.0/+4.5 μs")
    print(f"  初始失配 θ̂₀ = {THETA_INIT}；搜索盒 各轴 ±{HALF_BOX} 点"
          f"（{int((2 * HALF_BOX + 1) ** 3)} 个格）")
    print(f"  目标：f < {TARGET_DB:g} dB；预算 {MAX_EVALS} 次评估\n")

    rng = np.random.default_rng(seed)
    pts, vals = [], []

    def evaluate(u, tag=""):
        t0 = time.perf_counter()
        f = objective(u)
        dt = time.perf_counter() - t0
        pts.append(np.asarray(u, float))
        vals.append(f)
        th = to_theta(u)
        print(f"  [{len(vals):3d}] θ̂=({th['t_0']:2d},{th['T_L']:2d},{th['T_s']:2d})  "
              f"f = {f:7.2f} dB   ({dt:.1f} s)", flush=True)
        return f

    t_start = time.perf_counter()
    # ① 初始失配点（对应"搜索前"的抗干扰效果）
    u_init = (np.array([THETA_INIT[k] for k in KEYS], float) - LO) / (HI - LO)
    f_init = evaluate(u_init)
    if f_init < TARGET_DB:
        print("  初始估计已达标"); return
    # ② 其余初始点
    for _ in range(N_INIT_RAND):
        evaluate(rng.uniform(0, 1, 3))
    # ③ BO 主循环
    hit = False
    while len(vals) < budget:
        X, y = np.array(pts), np.array(vals)
        fmin = float(y.min())
        s2 = max(float(y.var()), 1e-3)
        # 候选只取栅格点（整数采样点参数），避免评估落在格内浪费
        n_lat = HI - LO
        cand = np.round(rng.uniform(0, 1, (N_CAND, 3)) * n_lat) / n_lat
        ib = int(y.argmin())
        loc = np.round((X[ib] + rng.normal(0, LOCAL_SIGMA, (N_LOCAL, 3)))
                       * (HI - LO)) / (HI - LO)
        cand = np.clip(np.vstack([cand, loc]), 0, 1)
        # 排除已评估过的格（局部扰动取整后会撞上旧点，白花一次评估）
        seen = {tuple(np.round(pt * n_lat).astype(int)) for pt in X}
        keep = np.array([tuple(np.round(c * n_lat).astype(int)) not in seen
                         for c in cand])
        cand = cand[keep]
        m, var = gp_predict(X, y, cand, s2, ELL)
        s = np.sqrt(var)
        zz = (fmin - m) / s
        ei = (fmin - m) * norm.cdf(zz) + s * norm.pdf(zz)
        u = cand[int(ei.argmax())]
        f_new = evaluate(u)
        if f_new < fmin:
            print("        ↑ 新最优", flush=True)
        if f_new < TARGET_DB:
            hit = True
            break
    elapsed = time.perf_counter() - t_start

    ib = int(np.argmin(vals))
    u_best, f_best = pts[ib], vals[ib]
    print(f"\n=== 结果 ===")
    print(f"  搜索耗时      : {elapsed:.1f} s（{elapsed / 60:.1f} min）")
    print(f"  评估次数      : {len(vals)} 次（首次为初始估计）")
    print(f"  搜索前电平    : {f_init:.2f} dB")
    print(f"  搜索后电平    : {f_best:.2f} dB   （改善 {f_init - f_best:.2f} dB）")
    print(f"  是否达标      : {'是' if hit else '否'}")
    print(f"  估计参数      : {to_theta(u_best)}   真实 {THETA_TRUE}")

    make_figure(THETA_INIT, to_theta(u_best), {
        "elapsed_s": elapsed, "n_evals": len(vals), "hit": bool(hit),
        "f_init_db": f_init, "f_best_db": f_best}, suffix)

    (OUTDIR / f"scene07_bo_search{suffix}.json").write_text(json.dumps({
        "theta_true": THETA_TRUE, "theta_init": THETA_INIT,
        "theta_best": to_theta(u_best), "box_half": HALF_BOX,
        "target_db": TARGET_DB, "max_evals": budget,
        "elapsed_s": elapsed, "n_evals": len(vals), "hit": bool(hit),
        "f_init_db": f_init, "f_best_db": f_best,
        "trace": [{"theta": to_theta(p), "f_db": float(v)}
                  for p, v in zip(pts, vals)],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"数据已保存：{OUTDIR / f'scene07_bo_search{suffix}.json'}")


def make_figure(theta_init: dict, theta_best: dict, stats: dict,
                suffix: str = "") -> None:
    """画"搜索前 / 搜索后"距离像，并把搜索时间等指标写在图下方中间。"""
    x_b, h_b = design_for(theta_init)
    x_a, h_a = design_for(theta_best)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), facecolor=SURFACE)
    win_km = (R_T_KM - 0.35, R_T_KM + 0.95)
    for ax, (xx, hh, title) in zip(axes, (
            (x_b, h_b, "Before search — initial estimate "
             + r"$\hat\theta_0$ = " + f"{list(theta_init.values())}"),
            (x_a, h_a, "After search — estimate "
             f"{list(theta_best.values())}"))):
        ax.set_facecolor(SURFACE)
        r_km, y_tot = scene_profile(xx, hh, J_TRUE, OFFS_TRUE, TAU_T)
        m = (r_km >= win_km[0]) & (r_km <= win_km[1])
        ax.plot(r_km[m], y_tot[m], color="#2a6f97", lw=1.0, zorder=2)
        ax.axvline(R_T_KM, color="#e4572e", lw=1.0, ls="--", zorder=3)
        for p in FTS_KM:
            lvl = float(y_tot[int(np.argmin(np.abs(r_km - p)))])
            ax.axvline(p, color="#c9a227", lw=0.8, ls=":", zorder=3)
            ax.scatter([p], [lvl], color="#c9a227", s=26, zorder=5,
                       edgecolor="white", linewidth=0.5)
            ax.annotate(f"{lvl:+.1f} dB", (p, lvl), xytext=(0, 5),
                        textcoords="offset points", ha="center", va="bottom",
                        fontsize=8, color="#8a6d1a", zorder=6)
        ax.set_ylim(-60, 6)
        ax.set_xlim(*win_km)
        ax.set_xlabel("Range (km)", color=TEXT_SECONDARY, fontsize=9)
        ax.set_title(title, color=TEXT_PRIMARY, fontsize=10.5)
        ax.tick_params(colors=TEXT_SECONDARY, labelsize=8.5)
        ax.grid(True, color="#e4e2dd", lw=0.4)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("Normalized level (dB)", color=TEXT_SECONDARY, fontsize=9)
    fig.suptitle("ISRJ parameter search by Bayesian optimization "
                 "(self-screening jamming, J/S = 10 dB, SNR = 10 dB)",
                 color=TEXT_PRIMARY, fontsize=12)
    hit_txt = "target reached" if stats["hit"] else "target NOT reached"
    fig.text(0.5, 0.015,
             f"Search time {stats['elapsed_s']:.1f} s "
             f"({stats['n_evals']} evaluations)    |    "
             f"level {stats['f_init_db']:.2f} → {stats['f_best_db']:.2f} dB "
             f"    |    criterion {TARGET_DB:g} dB: {hit_txt}",
             ha="center", va="bottom", fontsize=11, color=TEXT_PRIMARY)
    out = OUTDIR / f"scene07_bo_search_before_after{suffix}.png"
    fig.savefig(out, dpi=140, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"图已保存：{out}")


if __name__ == "__main__":
    main()
