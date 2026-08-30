"""Hermitian 正定线性方程组的公共求解工具。

模块三（Q）与模块四（A）都需要对 N×N Hermitian 矩阵做一次分解、多次回代。
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import LinAlgError, cho_factor, cho_solve


def hermitian_cholesky(A: np.ndarray, jitter: float = 1e-10):
    """Cholesky 分解 ``A``；接近奇异时按 PRD_03 §4.2 加对角正则化后重试。

    返回 :func:`scipy.linalg.cho_factor` 的分解结果。
    """
    A = 0.5 * (A + A.conj().T)
    scale = max(float(np.trace(A).real) / A.shape[0], 1.0)
    for k in range(6):
        try:
            return cho_factor(A, lower=True, check_finite=False)
        except LinAlgError:
            A = A + (jitter * 10 ** k * scale) * np.eye(A.shape[0])
    raise LinAlgError("matrix is not positive definite even after regularization")


def cho_solve_vec(factor, b: np.ndarray) -> np.ndarray:
    """用 :func:`hermitian_cholesky` 的分解结果求解 ``A v = b``。"""
    return cho_solve(factor, b, check_finite=False)
