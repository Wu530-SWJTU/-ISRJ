"""模块三：拉格朗日乘子法求解失配滤波器（PRD_03，论文式 7–8）。

固定波形 x 时，滤波器子问题是带双线性等式约束的凸二次规划：

    min_h  h^H Q h    s.t.  x^H h = b_max,  x_j^H h = b_min

闭式解为 ``Q^{-1}x`` 与 ``Q^{-1}x_j`` 的线性组合（论文式 (8) 印刷版漏掉了
第二项并漏了共轭，见 PRD_03 §3.4 勘误，此处采用数学正确的两项含共轭形式）。
"""

from __future__ import annotations

import numpy as np

from .numerics import cho_solve_vec, hermitian_cholesky


def solve_filter_lagrange(x: np.ndarray,
                          j: np.ndarray,
                          Q: np.ndarray,
                          b_max: float,
                          b_min: float) -> np.ndarray:
    """求解最优失配滤波器 h（解析解）。

    Parameters
    ----------
    x : shape=(N,) 复向量，当前波形
    j : shape=(N,) 实向量（0/1），ISRJ 采样函数
    Q : shape=(N,N) Hermitian 加权矩阵，必须由**当前** x 构造
    b_max, b_min : 主瓣峰值 / 峰值干扰约束（正实数）

    Returns
    -------
    h : shape=(N,) 复向量，满足 ``x^H h = b_max`` 与 ``x_j^H h = b_min``
    """
    x = np.asarray(x, dtype=complex)
    xj = x * np.asarray(j, dtype=float)

    factor = hermitian_cholesky(Q)
    q_x = cho_solve_vec(factor, x)        # Q^{-1} x
    q_xj = cho_solve_vec(factor, xj)      # Q^{-1} x_j

    a = np.vdot(x, q_x).real              # x^H Q^{-1} x   (实数)
    b = np.vdot(x, q_xj)                  # x^H Q^{-1} x_j (复数)
    c = np.vdot(xj, q_xj).real            # x_j^H Q^{-1} x_j (实数)

    det = a * c - abs(b) ** 2
    if det == 0 or not np.isfinite(det):
        raise np.linalg.LinAlgError(f"degenerate constraint system: det={det}")

    coeff1 = (b_max * c - b_min * b) / det
    coeff2 = (b_min * a - b_max * np.conjugate(b)) / det
    return coeff1 * q_x + coeff2 * q_xj


def filter_constraint_residuals(x: np.ndarray,
                                h: np.ndarray,
                                j: np.ndarray,
                                b_max: float,
                                b_min: float) -> tuple[float, float]:
    """返回两个约束的相对残差 ``(|x^H h - b_max|/b_max, |x_j^H h - b_min|/b_min)``。"""
    xj = np.asarray(x) * np.asarray(j, dtype=float)
    r_max = abs(np.vdot(x, h) - b_max) / abs(b_max)
    r_min = abs(np.vdot(xj, h) - b_min) / abs(b_min)
    return float(r_max), float(r_min)
