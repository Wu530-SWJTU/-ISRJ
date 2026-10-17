# -*- coding: utf-8 -*-
"""联合波形–失配滤波器设计（经典 ISRJ 重复转发，已知精确干扰）。

干扰模型（与 `run_scene_jamming.py` 场景一致）：
    信号级  y_jam = Σ_{r=1}^{R} shift_{r·T_L}( x ⊙ j )
    矩阵链  Y_jam = ( Σ_r S_{r·T_L} ) · diag(j) · x
            脉压输出端 H_j = H · ( Σ_r S_{r·T_L} ) · diag(j)
    假目标位置 +r·T_L（r = 1..R）

设计问题（把复现线 PRD 的交替优化从 2 条约束推广到 R+1 条）::

    min_h  h^H P h      s.t.  ⟨h, x⟩ = b_max,  ⟨h, v_r⟩ = b_min  (r = 1..R)
    min_x  x^H Q x      s.t.  同左（x 侧）+ PAPR 约束
    其中  v_r := Σ_{r'} S_{(r'−r)·T_L}(x ⊙ j)   （第 r 个假目标处的叠加约束向量）

**多份转发下 P、Q 不用改**（已推导）：
    Q 侧  X_{j,r}^H X_{j,r} = Toeplitz(r_{x⊙j})，移位不改变自相关；
    P 侧  P_r = diag(j)·S^H·T_h·S·diag(j)，Toeplitz 结构吸收移位 ⇒ P_r = P。
    真正区分 R 份转发的是**约束**：把干扰响应在 +r·T_L 处硬钉到 b_min。

求解器改动（仅此两处）：
    ① 滤波器闭式解：约束矩阵 C = [x, y_1, ..., y_R]（N×(R+1)），
       h = Q^{-1} C (C^H Q^{-1} C)^{-1} b，b = [b_max, b_min, ..., b_min]
    ② ADMM：约束向量 g_r := diag(j)·S_{r·T_L}^H h = j ⊙ (h 前移 r·T_L)，
       A = 2P + 2ρI + ρ h h^H + ρ Σ_r g_r g_r^H，R 个对偶变量 ν_r

用法：
    .venv/Scripts/python.exe 参数搜索/scripts/scene_joint_design.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import baseline  # noqa: E402
from isrj_joint_design.matrices import (  # noqa: E402
    assemble_P, assemble_Q, cross_correlation,
)
from isrj_joint_design.numerics import cho_solve_vec, hermitian_cholesky  # noqa: E402
from isrj_joint_design.params import generate_isrj_sampling_function  # noqa: E402
from scipy.linalg import toeplitz  # noqa: E402

OUTDIR = ROOT / "参数搜索" / "结果" / "对抗场景"

# ---- 场景/设计口径 ----
T_L_US = baseline.TAU_US            # 切片长度（唯一来源：baseline.py）
R_RETX = int(baseline.TS_US / baseline.TAU_US) - 1   # 占空比约束 (1+R)·T_L ≤ T_s
TAU_T_US = 30.0                     # 真目标时延（仅场景评估用）


def shift(v: np.ndarray, d: int) -> np.ndarray:
    """序列移位 d 个采样点（d > 0 延迟、d < 0 前移，越界补零截断）。"""
    if d == 0:
        return v.copy()
    out = np.zeros_like(v)
    if d > 0:
        out[d:] = v[:len(v) - d]
    else:
        out[:len(v) + d] = v[-d:]
    return out


def constraint_vectors(x: np.ndarray, j: np.ndarray, offsets: list,
                       lags: list) -> list:
    """各约束 lag 处的输出约束向量。

    干扰 J = Σ_r S_{d_r}(x⊙j) 在 lag ℓ 处的输出为
        Σ_r ⟨h, S_{d_r − ℓ}(x⊙j)⟩
    ⇒ 约束向量  v_ℓ = Σ_r S_{d_r − ℓ}(x ⊙ j)，约束 ⟨h, v_ℓ⟩ = b_min。
    约束 lag 取 {0} ∪ {假目标 lag}：ℓ=0 是真目标距离单元（干扰泄漏会扰动
    真目标幅度），ℓ=d_r 是各假目标位置。
    （单份且 ℓ=0 时退化为 v = x⊙j，即 PRD 的原式。）
    """
    xj = x * j
    return [sum((shift(xj, d - lag) for d in offsets), np.zeros_like(xj))
            for lag in lags]


def _autocorr_lags(v: np.ndarray) -> np.ndarray:
    """全自相关，按 lag 索引：返回 R 满足 ``R[τ + N − 1] = r_v[τ]``。"""
    n = len(v)
    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    rf = np.fft.ifft(np.abs(np.fft.fft(v, nfft)) ** 2)
    R = np.zeros(2 * n - 1, dtype=complex)
    R[n - 1:] = rf[:n]                    # 正 lag 0..N−1
    R[:n - 1] = rf[nfft - (n - 1):]       # 负 lag −(N−1)..−1
    return R


def _shift_sum_lags(R: np.ndarray, offsets: list) -> np.ndarray:
    """``q[τ] = Σ_{r,r'} r[τ + d_{r'} − d_r]``（移位和的自相关核）。"""
    m = len(R)
    q = np.zeros(m, dtype=complex)
    for dr in offsets:
        for drp_ in offsets:
            delta = int(drp_ - dr)
            lo, hi = max(0, -delta), min(m, m - delta)
            q[lo:hi] += R[lo + delta:hi + delta]
    return q


def _toeplitz_from_lags(R: np.ndarray) -> np.ndarray:
    """由 lag 核构造 Hermitian Toeplitz 矩阵。"""
    n = (len(R) + 1) // 2
    c = R[n - 1:]
    return toeplitz(c, np.conjugate(c))


def assemble_P_multi(h: np.ndarray, j: np.ndarray, offsets: list,
                     epsilon: float) -> np.ndarray:
    """多份转发下的 ``P``。

    实际干扰是 R 份移位列车之和 J = (Σ_r S_{d_r})(x⊙j)，其响应能量
    ``||H J||²`` 的 P 型矩阵为 ``diag(j)·A^H T_h A·diag(j)``，A = Σ_r S_{d_r}，
    而 ``(A^H T_h A)[i,j] = Σ_{r,r'} r_h[(i−j) + (d_{r'}−d_r)]`` 仍是 Toeplitz
    ⇒ 用移位和核组装，O(N²)。
    （单份时退化为 PRD 的 ``diag(j)·T_h·diag(j)``。）
    """
    h = np.asarray(h, dtype=complex)
    R = _shift_sum_lags(_autocorr_lags(h), offsets)
    T = _toeplitz_from_lags(R)
    T0 = _toeplitz_from_lags(_autocorr_lags(h))
    P = (epsilon * (T0 - np.outer(h, h.conj()))
         + (1.0 - epsilon) * (np.asarray(j, dtype=float)[:, None] * T
                              * np.asarray(j, dtype=float)[None, :]))
    return 0.5 * (P + P.conj().T)


def assemble_Q_multi(x: np.ndarray, j: np.ndarray, offsets: list,
                     epsilon: float) -> np.ndarray:
    """多份转发下的 ``Q``：``ε(T_x − xx^H) + (1−ε)·Toeplitz(r_u)``，

    ``u = Σ_r S_{d_r}(x ⊙ j)`` 为实际干扰信号（其自相关核同样由移位和得到）。
    """
    x = np.asarray(x, dtype=complex)
    R = _shift_sum_lags(_autocorr_lags(x * np.asarray(j, dtype=float)), offsets)
    T = _toeplitz_from_lags(R)
    T0 = _toeplitz_from_lags(_autocorr_lags(x))
    Q = (epsilon * (T0 - np.outer(x, x.conj()))
         + (1.0 - epsilon) * T)
    return 0.5 * (Q + Q.conj().T)


def constraint_functionals(h: np.ndarray, j: np.ndarray, offsets: list,
                           lags: list) -> list:
    """x 空间的约束向量 g_ℓ，满足 ⟨h, v_ℓ⟩ = ⟨g_ℓ, x⟩。"""
    return [j * sum((shift(h, lag - d) for d in offsets), np.zeros_like(h))
            for lag in lags]


def solve_filter_multi(x: np.ndarray, j: np.ndarray, offsets: list, lags: list,
                       Q: np.ndarray, b_max: float, b_min: float) -> np.ndarray:
    """R+1 条等式约束下的闭式失配滤波器：

    min_h h^H Q h  s.t.  ⟨h,x⟩ = b_max, ⟨h,v_r⟩ = b_min (r = 1..R)
    ⇒ h = Q^{-1} C (C^H Q^{-1} C)^{-1} b，C = [x, v_1, ..., v_R]
    """
    factor = hermitian_cholesky(Q)
    cols = [np.asarray(x, dtype=complex)] + constraint_vectors(x, j, offsets, lags)
    C = np.column_stack(cols)                       # N × (R+1)
    QiC = np.column_stack([cho_solve_vec(factor, C[:, k])
                           for k in range(C.shape[1])])
    G = C.conj().T @ QiC                            # (R+1)×(R+1) Hermitian
    b = np.array([b_max] + [b_min] * len(lags), dtype=complex)
    coeff = np.linalg.solve(G, b)
    return QiC @ coeff


def solve_waveform_admm_multi(h: np.ndarray, j: np.ndarray, offsets: list,
                              lags: list,
                              P: np.ndarray, params, variables: dict
                              ) -> tuple[np.ndarray, dict, dict]:
    """ADMM 解波形子问题（R+1 条约束版）。

    与复现线 `solve_waveform_admm` 的唯一区别：干扰侧约束由 1 条变 R 条，
    对偶变量 nu 由标量变 R 维向量，A 中多出 Σ_r ρ g_r g_r^H。
    """
    N = params.N
    rho = params.rho
    b_max, b_min = params.b_max, params.b_min
    sqrt_eta = np.sqrt(params.eta)

    h = np.asarray(h, dtype=complex)
    g_list = constraint_functionals(h, j, offsets, lags)

    x = variables["x"].copy()
    y = variables["y"].copy()
    z = variables["z"].copy()
    lam = variables["lambda"].copy()
    gam = variables["gamma"].copy()
    mu = complex(variables["mu"])
    nu_vec = np.asarray(variables["nu_vec"], dtype=complex).copy()

    A = 2.0 * P + 2.0 * rho * np.eye(N) + rho * np.outer(h, h.conj())
    for g in g_list:
        A += rho * np.outer(g, g.conj())
    factor = hermitian_cholesky(A)

    consensus_error = np.inf
    iterations = 0
    for iterations in range(1, params.max_inner_iter + 1):
        b_vec = rho * (z - lam) + rho * (y - gam) + rho * (b_max - mu) * h
        for g, nu in zip(g_list, nu_vec):
            b_vec = b_vec + rho * (b_min - nu) * g
        x = cho_solve_vec(factor, b_vec)

        w_y = x + gam
        norm_y = np.linalg.norm(w_y)
        if norm_y < 1e-12:
            w_y = np.exp(2j * np.pi * np.random.default_rng(iterations).random(N))
            norm_y = np.linalg.norm(w_y)
        y = np.sqrt(N) * w_y / norm_y

        w_z = x + lam
        abs_wz = np.abs(w_z)
        z = np.where(abs_wz <= sqrt_eta, w_z, sqrt_eta * w_z / (abs_wz + 1e-12))

        lam = x - z + lam
        gam = x - y + gam
        mu = np.vdot(h, x) - b_max + mu
        nu_vec = np.array([np.vdot(g, x) - b_min + nu
                           for g, nu in zip(g_list, nu_vec)])

        consensus_error = float(np.linalg.norm(x - y) + np.linalg.norm(x - z))
        if consensus_error < params.delta:
            break

    variables_updated = {
        "x": x, "y": y, "z": z, "lambda": lam, "gamma": gam,
        "mu": mu, "nu_vec": nu_vec, "h": variables.get("h"),
    }
    info = {"iterations": iterations, "consensus_error": consensus_error,
            "converged": consensus_error < params.delta}
    return x, variables_updated, info


def joint_design_scene(params, j: np.ndarray, offsets: list,
                       lags: list, verbose: bool = True) -> dict:
    """交替优化主循环（R 份转发版）。"""
    N = params.N
    x = np.exp(1j * 2 * np.pi * np.random.default_rng(params.seed).random(N))
    h = np.zeros(N, dtype=complex)
    # 热启动状态：nu 扩展为 R 维
    rng = np.random.default_rng(params.seed)
    variables = {
        "x": x.copy(), "y": x.copy(), "z": x.copy(),
        "lambda": np.zeros(N, dtype=complex), "gamma": np.zeros(N, dtype=complex),
        "mu": 0j, "nu_vec": np.zeros(len(lags), dtype=complex),
        "h": h.copy(),
    }
    hist = {"isl": [], "il": [], "loss": [], "admm_iters": []}
    t0 = time.perf_counter()
    for m in range(1, params.M + 1):
        Q = assemble_Q_multi(x, j, offsets, params.epsilon)
        h = solve_filter_multi(x, j, offsets, lags, Q,
                               params.b_max, params.b_min)

        P = assemble_P_multi(h, j, offsets, params.epsilon)
        variables["x"] = x
        variables["h"] = h
        x, variables, info = solve_waveform_admm_multi(h, j, offsets, lags,
                                                       P, params, variables)
        v_r = constraint_vectors(x, j, offsets, lags)
        b_axis = np.abs(np.vdot(h, x))
        il_db = 20 * np.log10(max(abs(np.vdot(h, vr)) for vr in v_r) / b_axis)
        isl = float(np.sum(np.abs(cross_correlation(x, h)) ** 2)
                    - abs(np.vdot(h, x)) ** 2)
        hist["isl"].append(isl)
        hist["il"].append(il_db)
        hist["admm_iters"].append(info["iterations"])
        if verbose and (m % 20 == 0 or m == 1 or m == params.M):
            print(f"  [{m:3d}/{params.M}] 干扰峰值 {il_db:7.2f} dB  "
                  f"admm_iter={info['iterations']:3d}  "
                  f"cons={info['consensus_error']:.2e}", flush=True)

    elapsed = time.perf_counter() - t0
    v_r = constraint_vectors(x, j, offsets, lags)
    b_axis = float(abs(np.vdot(h, x)))
    return {
        "x": x, "h": h, "j": j, "offsets": offsets, "lags": lags,
        "elapsed_s": elapsed,
        "b_axis": b_axis,
        "il_db": [float(20 * np.log10(abs(np.vdot(h, vr)) / b_axis)) for vr in v_r],
        "residual_b_max": float(abs(np.vdot(h, x) - params.b_max) / params.b_max),
        "residual_b_min": [float(abs(abs(np.vdot(h, vr)) - params.b_min) / params.b_min)
                           for vr in v_r],
        "history": hist,
    }
