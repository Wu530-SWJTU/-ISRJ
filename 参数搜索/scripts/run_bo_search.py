"""在插值图上跑一次 BO 搜索（W3 定标口径），打印完整搜索轨迹。

口径（`docs/工作流.md` W3，用户 2026-09-26 定）：
    目标函数 = `interp_t0_ts.py` 的双线性插值 f(t_0, T_s)（dB）
    成功判据 = 找到任意 f < -60 dB 的参数点即停；估计到什么用什么，不取整
    预算     = --budget 次评估上限（在线搜索要短，默认 20）

BO 配置（实测定标）：GP 核 Matern 3/2、ℓ = 0.1（归一化 = 1 采样点）、
σ² = 观测值样本方差、初始 4 个盒内随机点、每轮 2000 均匀候选 + 200 高斯扰动、
采集函数 EI（最小化版）。

用法：
    .venv/Scripts/python.exe 参数搜索/scripts/run_bo_search.py --seed 0 --budget 20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from interp_t0_ts import load_interpolator  # noqa: E402

T0_RANGE = (1.5, 2.5)      # 搜索盒（实际值 μs）
TS_RANGE = (5.9, 6.9)
SUCCESS_DB = -60.0         # 成功判据（用户 2026-09-26 定）
ELL = 0.1                  # GP 核长度尺度（归一化 = 1 采样点）
N_INIT = 4                 # 初始随机点数
N_CAND = 2000              # 每轮均匀候选数
N_LOCAL = 200              # 每轮围绕当前最优的高斯扰动数
LOCAL_SIGMA = 0.05         # 扰动 σ（归一化）

f = load_interpolator()


def z(u):
    t0 = T0_RANGE[0] + u[0] * (T0_RANGE[1] - T0_RANGE[0])
    ts = TS_RANGE[0] + u[1] * (TS_RANGE[1] - TS_RANGE[0])
    return float(f(t0, ts))


def unbox(u):
    return (T0_RANGE[0] + u[0] * (T0_RANGE[1] - T0_RANGE[0]),
            TS_RANGE[0] + u[1] * (TS_RANGE[1] - TS_RANGE[0]))


def mat32(X, Y, s2, ell):
    d = np.sqrt(((X[:, None, :] - Y[None, :, :]) ** 2).sum(-1))
    return s2 * (1 + np.sqrt(3) * d / ell) * np.exp(-np.sqrt(3) * d / ell)


def gp_predict(Xo, yo, Xn, s2, ell, jitter=1e-9):
    mu = yo.mean()
    K = mat32(Xo, Xo, s2, ell) + jitter * np.eye(len(Xo))
    c, low = cho_factor(K)
    a = cho_solve((c, low), yo - mu)
    Ks = mat32(Xo, Xn, s2, ell)
    m = mu + Ks.T @ a
    v = cho_solve((c, low), Ks)
    var = s2 - (Ks * v).sum(0)          # 自核对角 = σ²，无需拼 2200×2200 矩阵
    return m, np.clip(var, 1e-12, None)


def _wrap(seed, pts, lvls, hit, verbose):
    ib = int(np.argmin(lvls))
    t0, ts = unbox(pts[ib])
    if verbose:
        print(f"  >>> {'命中' if hit else '预算用完，未命中'}：共 {len(lvls)} 次评估；"
              f"最优 f = {lvls[ib]:.2f} dB @ t0 = {t0:.4f}、ts = {ts:.4f}")
    return {"seed": seed, "evals": len(lvls), "hit": hit,
            "best": (float(t0), float(ts), float(lvls[ib])),
            "trace": [(unbox(u)[0], unbox(u)[1], float(l)) for u, l in zip(pts, lvls)]}


def bo_search(seed: int, budget: int = 20, verbose: bool = True) -> dict:
    rng = np.random.default_rng(seed)
    pts, lvls = [], []
    if verbose:
        print(f"=== BO 搜索（seed {seed}，预算 {budget} 次评估，判据 f < {SUCCESS_DB:g} dB）===")
        print(f"  搜索盒：t0 ∈ [{T0_RANGE[0]}, {T0_RANGE[1]}] μs、"
              f"T_s ∈ [{TS_RANGE[0]}, {TS_RANGE[1]}] μs")
    for k in range(N_INIT):
        u = rng.uniform(0, 1, 2)
        pts.append(u)
        lvls.append(z(u))
        if verbose:
            t0, ts = unbox(u)
            print(f"  初始点 {k + 1}   t0 = {t0:.4f}   ts = {ts:.4f}   f = {lvls[-1]:+8.2f} dB")
    if min(lvls) < SUCCESS_DB:
        return _wrap(seed, pts, lvls, True, verbose)
    for it in range(1, budget - N_INIT + 1):
        X, y = np.array(pts), np.array(lvls)
        fmin = float(y.min())
        s2 = max(float(y.var()), 1e-3)
        cand = rng.uniform(0, 1, (N_CAND, 2))
        ib = int(y.argmin())
        cand = np.vstack([cand, X[ib] + rng.normal(0, LOCAL_SIGMA, (N_LOCAL, 2))])
        cand = np.clip(cand, 0, 1)
        m, var = gp_predict(X, y, cand, s2, ELL)
        s = np.sqrt(var)
        zz = (fmin - m) / s
        ei = (fmin - m) * norm.cdf(zz) + s * norm.pdf(zz)
        u = cand[int(ei.argmax())]
        pts.append(u)
        lvls.append(z(u))
        if verbose:
            t0, ts = unbox(u)
            flag = "   <- 新最优" if lvls[-1] <= fmin else ""
            print(f"  第 {it:2d} 次   t0 = {t0:.4f}   ts = {ts:.4f}   f = {lvls[-1]:+8.2f} dB{flag}")
        if lvls[-1] < SUCCESS_DB:
            return _wrap(seed, pts, lvls, True, verbose)
    return _wrap(seed, pts, lvls, False, verbose)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget", type=int, default=20,
                    help="评估次数上限（在线搜索要短，默认 20）")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    bo_search(args.seed, args.budget, verbose=True)


if __name__ == "__main__":
    main()
