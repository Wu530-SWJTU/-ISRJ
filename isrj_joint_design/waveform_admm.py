"""模块四：ADMM 求解波形子问题（PRD_04，论文式 9–16）。

固定失配滤波器 h 时，波形子问题因 PAPR / 球面约束而非凸。引入辅助变量
``y``（承担 ``y^H y = N``）与 ``z``（承担 ``|z[n]|^2 <= eta``）后，四个 ADMM
子步骤全部有闭式解：

    x <- A^{-1} b_vec,  A = 2P + 2rho I + rho h h^H + rho h_j h_j^H   (式 14)
    y <- sqrt(N) (x+gamma)/||x+gamma||                                 (式 15)
    z <- 逐元素把模长裁剪到 sqrt(eta)，相位不变                        (式 16)
    对偶变量按 scaled form 更新                                        (式 13d)
"""

from __future__ import annotations

import numpy as np

from .numerics import cho_solve_vec, hermitian_cholesky
from .params import GlobalParams


def solve_waveform_admm(h: np.ndarray,
                        j: np.ndarray,
                        P: np.ndarray,
                        params: GlobalParams,
                        variables: dict,
                        record_objective: bool = False) -> tuple[np.ndarray, dict, dict]:
    """ADMM 求解 ``min x^H P x`` s.t. 主瓣/干扰峰值约束 + PAPR 约束。

    Parameters
    ----------
    h : shape=(N,) 固定的失配滤波器
    j : shape=(N,) ISRJ 采样函数
    P : shape=(N,N) 由**当前** h 构造的加权矩阵
    params : 全局参数（rho, delta, eta, b_max, b_min, max_inner_iter）
    variables : 含 x, y, z, lambda, gamma, mu, nu 的热启动状态（不就地修改）
    record_objective : 是否记录每次迭代的 ``x^H P x``（调试/验收用）

    Returns
    -------
    (x_new, variables_updated, info)
    """
    N = params.N
    rho = params.rho
    b_max, b_min = params.b_max, params.b_min
    sqrt_eta = np.sqrt(params.eta)

    h = np.asarray(h, dtype=complex)
    h_j = h * np.asarray(j, dtype=float)

    x = variables["x"].copy()
    y = variables["y"].copy()
    z = variables["z"].copy()
    lam = variables["lambda"].copy()
    gam = variables["gamma"].copy()
    mu = complex(variables["mu"])
    nu = complex(variables["nu"])

    # 内层迭代中 h 固定 => A 不变，只需分解一次（式 14）
    A = (2.0 * P + 2.0 * rho * np.eye(N)
         + rho * np.outer(h, h.conj()) + rho * np.outer(h_j, h_j.conj()))
    factor = hermitian_cholesky(A)

    objective = []
    consensus_error = np.inf
    iterations = 0
    for iterations in range(1, params.max_inner_iter + 1):
        # 步骤 1：更新 x（式 13a / 14）
        b_vec = (rho * (z - lam) + rho * (y - gam)
                 + rho * (b_max - mu) * h + rho * (b_min - nu) * h_j)
        x = cho_solve_vec(factor, b_vec)

        # 步骤 2：更新 y —— 投影到半径 sqrt(N) 的复球面（式 13b / 15）
        w_y = x + gam
        norm_y = np.linalg.norm(w_y)
        if norm_y < 1e-12:                      # 数值下溢时的兜底（PRD_04 §4.2）
            w_y = np.exp(2j * np.pi * np.random.default_rng(iterations).random(N))
            norm_y = np.linalg.norm(w_y)
        y = np.sqrt(N) * w_y / norm_y

        # 步骤 3：更新 z —— 逐元素模长裁剪，相位不变（式 13c / 16）
        w_z = x + lam
        abs_wz = np.abs(w_z)
        z = np.where(abs_wz <= sqrt_eta, w_z, sqrt_eta * w_z / (abs_wz + 1e-12))

        # 步骤 4：更新对偶变量（式 13d）
        lam = x - z + lam
        gam = x - y + gam
        mu = np.vdot(h, x) - b_max + mu
        nu = np.vdot(h_j, x) - b_min + nu

        if record_objective:
            objective.append(float(np.vdot(x, P @ x).real))

        # 收敛判据：两个 2-范数之和（不是平方和）
        consensus_error = float(np.linalg.norm(x - y) + np.linalg.norm(x - z))
        if consensus_error < params.delta:
            break

    variables_updated = {
        "x": x, "y": y, "z": z,
        "lambda": lam, "gamma": gam, "mu": mu, "nu": nu,
        "h": variables.get("h"),
    }
    info = {
        "iterations": iterations,
        "consensus_error": consensus_error,
        "converged": consensus_error < params.delta,
        "objective": objective,
    }
    return x, variables_updated, info
