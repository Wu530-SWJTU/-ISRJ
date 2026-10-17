# -*- coding: utf-8 -*-
"""对抗场景：WJPS 展宽的完整 (x, h) 联合设计（已定稿口径，2026-10-17）。

口径（见 docs/adr/0005–0008）
    度量     假目标 lag 上的电平（取法 1）
    电平块   逐档平方和（ADR-0006）
    归一化   κ = trace(E)/trace(L)，固定；比例旋钮 γ（ADR-0007），默认 0.3
    约束     复合掩膜 ĵ = Σ_m ω_m j_m（ADR-0008）
    b_min    −50 dB（ADR-0009，场景展宽专用；`baseline.py` 的全局基线仍是 −80）

干扰模型
    信号级   y_jam(θ) = Σ_r shift_{d_r(θ)}( x ⊙ j(θ) ),   d_r = r·T_L(θ)
    矩阵链   H_{j,m} = H·diag(j_m)
             E_m     = Toeplitz(q_m),  q_m[τ] = Σ_{r,r'} r_{x⊙j_m}[τ + d_{r'} − d_r]
             v_{ℓ,m} = Σ_r shift(d_r − ℓ)( x ⊙ j_m )

目标函数
    Q = ε·(T_x − x x^H) + (1−ε)·[ (1−γ)·E_0 + γ·L̂_Q ]
    P = ε·(T_h − h h^H) + (1−ε)·[ diag(j_0)·T_h·diag(j_0) + γ·L̂_P ]
    L_Q = Σ_m ω_m Σ_{ℓ∈假目标} v_{ℓ,m} v_{ℓ,m}^H
    L_P = Σ_m ω_m Σ_{ℓ∈假目标} g_{ℓ,m} g_{ℓ,m}^H ,  g_{ℓ,m} = j_m ⊙ Σ_r shift(ℓ − d_r)(h)
    L̂   = L × trace(E)/trace(L)          ← 每轮用当前 x/h 重算
    ω_m ∝ 1 + |Δ_m|                      ← 3:2:1:2:3（误差越大权重越大）

约束（复合）
    ⟨h, x⟩ = b_max,  ⟨h, v_ℓ(ĵ)⟩ = b_min,  ℓ ∈ {0} ∪ 假目标 lag,  ĵ = Σ_m ω_m j_m

用法
    .venv/Scripts/python.exe 参数搜索/scripts/run_scene_wjps.py [--gamma 0.3] [--m-outer 100]
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
from scene_joint_design import (  # noqa: E402
    _autocorr_lags, _shift_sum_lags, _toeplitz_from_lags,
    constraint_functionals, constraint_vectors, joint_design_scene,
    solve_filter_multi, solve_waveform_admm_multi,
)

OUTDIR = ROOT / "参数搜索" / "结果" / "对抗场景"

BASE = {
    "t_0": int(round(baseline.T0_US / baseline.SAMPLE_US)),
    "T_L": int(round(baseline.TAU_US / baseline.SAMPLE_US)),
    "T_s": int(round(baseline.TS_US / baseline.SAMPLE_US)),
}
N_SLICES = 4
R_MAX = BASE["T_s"] // BASE["T_L"] - 1
DEL = (-2, -1, 0, 1, 2)          # T_s 误差档（±0.2 μs）
ORDER = (0, -2, -1, 1, 2)        # 零误差档必须第一（设计锚点）


def build_mask(n_off: int, n_tau: int, n_ts: int, N: int = 256) -> np.ndarray:
    j = np.zeros(N)
    for s in range(N_SLICES):
        st = n_off + s * n_ts
        if st >= N:
            break
        j[st:min(st + n_tau, N)] = 1.0
    return j


def n_retx(n_ts: int) -> int:
    return max(min(R_MAX, n_ts // BASE["T_L"] - 1), 1)


def family() -> list:
    """[(掩膜, 转发偏移, 权重)]，零误差档排第一。"""
    W = {e: 1 + abs(e) for e in DEL}
    s = sum(W.values())
    out = []
    for e in ORDER:
        ts = BASE["T_s"] + e
        out.append((build_mask(BASE["t_0"], BASE["T_L"], ts),
                    [r * BASE["T_L"] for r in range(1, n_retx(ts) + 1)],
                    W[e] / s))
    return out


def level_block_Q(x: np.ndarray, fam: list, E: np.ndarray) -> np.ndarray:
    L = np.zeros((len(x), len(x)), dtype=complex)
    for j, offs, w in fam:
        for v in constraint_vectors(x, j, offs, [0] + offs)[1:]:
            L += w * np.outer(v, v.conj())
    L /= len(fam)
    return L * (np.trace(E).real / np.trace(L).real)


def level_block_P(h: np.ndarray, fam: list, E: np.ndarray) -> np.ndarray:
    L = np.zeros((len(h), len(h)), dtype=complex)
    for j, offs, w in fam:
        for g in constraint_functionals(h, j, offs, offs):
            L += w * np.outer(g, g.conj())
    L /= len(fam)
    return L * (np.trace(E).real / np.trace(L).real)


def joint_design_wjps(params, fam, j_hat: np.ndarray, gamma: float,
                     verbose: bool = True) -> dict:
    N = params.N
    j0, offs0, _ = fam[0]
    j0f = np.asarray(j0, dtype=float)
    lags = [0] + offs0
    x = np.exp(1j * 2 * np.pi * np.random.default_rng(params.seed).random(N))
    h = np.zeros(N, dtype=complex)
    var = {"x": x.copy(), "y": x.copy(), "z": x.copy(),
           "lambda": np.zeros(N, dtype=complex), "gamma": np.zeros(N, dtype=complex),
           "mu": 0j, "nu_vec": np.zeros(len(lags), dtype=complex), "h": h.copy()}
    t0 = time.perf_counter()
    for m in range(1, params.M + 1):
        E_Q = _toeplitz_from_lags(_shift_sum_lags(_autocorr_lags(x * j0), offs0))
        T_isl = _toeplitz_from_lags(_autocorr_lags(x)) - np.outer(x, x.conj())
        Q = params.epsilon * T_isl + (1 - params.epsilon) * (
            (1 - gamma) * E_Q + gamma * level_block_Q(x, fam, E_Q))
        Q = 0.5 * (Q + Q.conj().T)
        h = solve_filter_multi(x, j_hat, offs0, lags, Q, params.b_max, params.b_min)

        E_P = j0f[:, None] * _toeplitz_from_lags(
            _shift_sum_lags(_autocorr_lags(h), offs0)) * j0f[None, :]
        T_h = _toeplitz_from_lags(_autocorr_lags(h)) - np.outer(h, h.conj())
        P = params.epsilon * T_h + (1 - params.epsilon) * (
            (1 - gamma) * E_P + gamma * level_block_P(h, fam, E_P))
        P = 0.5 * (P + P.conj().T)

        var["x"], var["h"] = x, h
        x, var, info = solve_waveform_admm_multi(
            h, j_hat, offs0, lags, P, params, var)
        if verbose and (m % 20 == 0 or m == 1 or m == params.M):
            print(f"  [{m:3d}/{params.M}] admm_iter={info['iterations']:3d}  "
                  f"cons={info['consensus_error']:.2e}", flush=True)
    return {"x": x, "h": h, "elapsed_s": time.perf_counter() - t0}


def measure(x: np.ndarray, h: np.ndarray) -> dict:
    N = len(x)
    pk = abs(np.vdot(h, x))
    levels = []
    for e in range(-5, 6):
        ts = BASE["T_s"] + e
        offs = [r * BASE["T_L"] for r in range(1, n_retx(ts) + 1)]
        v = constraint_vectors(x, build_mask(BASE["t_0"], BASE["T_L"], ts),
                               offs, [0] + offs)
        levels.append(max(20 * np.log10(abs(np.vdot(h, vv)) / pk) for vv in v[1:]))
    r = np.convolve(x, np.conj(h[::-1]), mode="full")
    psl = 20 * np.log10(np.delete(np.abs(r), N - 1).max() / pk)
    lpg = 20 * np.log10(pk / np.sqrt(N * np.sum(np.abs(h) ** 2)))
    return {"levels_db": [float(v) for v in levels],
            "psl_db": float(psl), "lpg_db": float(lpg)}


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gamma", type=float, default=0.3)
    ap.add_argument("--bmin", type=float, default=-50.0)
    ap.add_argument("--m-outer", type=int, default=100)
    args = ap.parse_args()

    params = dataclasses.replace(
        baseline.params_for(peak_interf_db=args.bmin), M=args.m_outer)
    fam = family()
    j0, offs0, _ = fam[0]
    j_hat = sum(w * j for j, _, w in fam)
    print(f"b_min = {args.bmin:g} dB，γ = {args.gamma:g}，"
          f"M_outer = {params.M}，复合约束")
    print("权重 ω = " + ", ".join(f"{w:.3f}" for _, _, w in fam) + "\n")

    # 扩V前：现有设计，在同一个 b_min 下重新设计（保证前后口径一致）
    t = time.perf_counter()
    base = joint_design_scene(params, j0, offs0, [0] + offs0, verbose=False)
    print(f"扩V前设计完成：{time.perf_counter() - t:.1f} s")

    t = time.perf_counter()
    res = joint_design_wjps(params, fam, j_hat, args.gamma, verbose=True)
    print(f"扩V后设计完成：{time.perf_counter() - t:.1f} s\n")

    before = measure(base["x"], base["h"])
    after = measure(res["x"], res["h"])
    xs = list(range(-5, 6))
    print(f"{'误差(采样点)':>12}" + "".join(f"{e:+7d}" for e in xs))
    for tag, r in (("扩V前", before), ("扩V后", after)):
        print(f"{tag:>12}" + "".join(f"{v:7.1f}" for v in r["levels_db"])
              + f"   | PSL {r['psl_db']:6.2f}  LPG {r['lpg_db']:5.2f}")

    tag = f"gamma{args.gamma:g}_bmin{abs(args.bmin):g}dB"
    np.savez_compressed(OUTDIR / f"scene_wjps_design_{tag}.npz",
                        x=res["x"], h=res["h"], x_before=base["x"],
                        h_before=base["h"], j_hat=j_hat)
    payload = {"gamma": args.gamma, "b_min_db": args.bmin,
               "m_outer": params.M, "constraint": "composite",
               "family": "T_s ±2 samples", "before": before, "after": after,
               "elapsed_s": res["elapsed_s"]}
    out = OUTDIR / f"scene08_composite_{tag}.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n数据已保存：{out.name}")
    print(f"设计对已保存：scene_wjps_design_{tag}.npz")


if __name__ == "__main__":
    main()
