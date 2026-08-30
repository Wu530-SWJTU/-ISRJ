"""模块二：矩阵构造与目标函数加权矩阵（PRD_02）。

共轭约定（重要）
----------------
论文/PRD 把主瓣峰值写成 ``h^H x``（模块四）与 ``x^H h``（模块三），两者对实
数 ``b_max`` 等价。要让卷积输出的第 N 个元素**恰好**等于这个带共轭的内积，
矩阵必须由**共轭后的**序列铺成：

    H[k, n] = conj(h[k - N + n])        =>  (H x)[N] = h^H x
    X[k, n] = conj(x[N + n - k])        =>  (X h)[N] = x^H h

于是 ``r = H x`` 与 ``r_x = X h`` 互为共轭（``X h = conj(H x)``），模长完全
相同，因此 ``x^H P x == h^H Q h`` 严格成立，而 PRD_02 §7 验收 1 的
"H x 与 X h 一致"在本实现中表现为"共轭一致"。这是 PRD_02 §2.2/§2.3 的图示
（未写共轭）与 §7 验收 2（写了 ``h^H x``）之间唯一自洽的取法：``H x`` 关于
``h`` 是共轭线性的，不可能同时等于一个关于 ``h`` 线性的 ``X h``。

只有本模块的矩阵元素定义带这个 conj；模块三/四/五的公式与 PRD 原文逐字一致。
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import toeplitz

__all__ = [
    "build_convolution_matrix",
    "build_correlation_matrix",
    "build_isrj_convolution_matrix",
    "build_isrj_correlation_matrix",
    "build_D_matrix",
    "build_P",
    "build_Q",
    "autocorrelation_matrix",
    "assemble_P",
    "assemble_Q",
    "cross_correlation",
]


# --------------------------------------------------------------------------
# 显式矩阵构造（PRD_02 §2–§4 的直译实现，O(N^2) 内存，用于验证与小规模场景）
# --------------------------------------------------------------------------

def build_convolution_matrix(h: np.ndarray, N: int) -> np.ndarray:
    """构造滤波器卷积矩阵 H，shape=(2N-1, N)，满足 ``(H x)[N-1] = h^H x``。"""
    hc = np.conjugate(np.asarray(h, dtype=complex))
    H = np.zeros((2 * N - 1, N), dtype=complex)
    for n in range(N):                       # 0-indexed: H[k, n] = conj(h[k - N + 1 + n])
        H[N - 1 - n:2 * N - 1 - n, n] = hc
    return H


def build_correlation_matrix(x: np.ndarray, N: int) -> np.ndarray:
    """构造波形互相关矩阵 X，shape=(2N-1, N)，满足 ``(X h)[N-1] = x^H h``。"""
    xc = np.conjugate(np.asarray(x, dtype=complex))
    X = np.zeros((2 * N - 1, N), dtype=complex)
    for n in range(N):                       # 0-indexed: X[k, n] = conj(x[N - 1 + n - k])
        X[n:n + N, n] = xc[::-1]
    return X


def build_isrj_convolution_matrix(h: np.ndarray, j: np.ndarray, N: int) -> np.ndarray:
    """构造 ISRJ 卷积矩阵 ``H_j = H diag(j)``（列方向乘采样函数）。"""
    return build_convolution_matrix(h, N) * np.asarray(j, dtype=float)[None, :]


def build_isrj_correlation_matrix(x: np.ndarray, j: np.ndarray, N: int) -> np.ndarray:
    """构造 ISRJ 互相关矩阵 X_j（由 ``x_j = x ⊙ j`` 铺成，不是 X diag(j)）。"""
    return build_correlation_matrix(np.asarray(x) * np.asarray(j), N)


def build_D_matrix(N: int) -> np.ndarray:
    """主瓣屏蔽对角矩阵 D，shape=(2N-1, 2N-1)，仅第 N 个对角元为 0。"""
    d = np.ones(2 * N - 1)
    d[N - 1] = 0.0
    return np.diag(d)


def build_P(H: np.ndarray, Hj: np.ndarray, D: np.ndarray, epsilon: float) -> np.ndarray:
    """``P = eps * H^H D H + (1-eps) * Hj^H Hj``（PRD_02 §4.1）。"""
    P = epsilon * (H.conj().T @ D @ H) + (1.0 - epsilon) * (Hj.conj().T @ Hj)
    return 0.5 * (P + P.conj().T)


def build_Q(X: np.ndarray, Xj: np.ndarray, D: np.ndarray, epsilon: float) -> np.ndarray:
    """``Q = eps * X^H D X + (1-eps) * Xj^H Xj``（PRD_02 §4.2）。"""
    Q = epsilon * (X.conj().T @ D @ X) + (1.0 - epsilon) * (Xj.conj().T @ Xj)
    return 0.5 * (Q + Q.conj().T)


# --------------------------------------------------------------------------
# 快速组装（PRD_02 §4.3）：利用 Toeplitz 结构，避开 (2N-1)×N 大矩阵
# --------------------------------------------------------------------------

def autocorrelation_matrix(v: np.ndarray) -> np.ndarray:
    """``H^H H``（或 ``X^H X``）的 Toeplitz 形式，元素为 ``r_v[i-j]``。

    ``r_v[d] = sum_m conj(v[m]) v[m+d]``，通过 FFT 以 O(N log N) 计算。
    """
    v = np.asarray(v, dtype=complex)
    N = v.size
    n_fft = 1 << int(np.ceil(np.log2(2 * N)))
    V = np.fft.fft(v, n_fft)
    r = np.fft.ifft(np.abs(V) ** 2)[:N]          # r[d] = sum_m conj(v[m]) v[m+d]
    return toeplitz(r, np.conjugate(r))          # Hermitian: M[i,j] = r[i-j]


def assemble_P(h: np.ndarray, j: np.ndarray, epsilon: float) -> np.ndarray:
    """快速组装 ``P``，等价于 :func:`build_P`。

    ``H^H D H = H^H H - h h^H``，``H_j^H H_j = diag(j) (H^H H) diag(j)``。
    """
    h = np.asarray(h, dtype=complex)
    j = np.asarray(j, dtype=float)
    T = autocorrelation_matrix(h)
    P = epsilon * (T - np.outer(h, h.conj())) + (1.0 - epsilon) * (T * np.outer(j, j))
    return 0.5 * (P + P.conj().T)


def assemble_Q(x: np.ndarray, j: np.ndarray, epsilon: float) -> np.ndarray:
    """快速组装 ``Q``，等价于 :func:`build_Q`。

    ``X^H D X = X^H X - x x^H``；``X_j^H X_j`` 是 ``x ⊙ j`` 的自相关 Toeplitz 阵。
    """
    x = np.asarray(x, dtype=complex)
    xj = x * np.asarray(j, dtype=float)
    T = autocorrelation_matrix(x)
    Q = epsilon * (T - np.outer(x, x.conj())) + (1.0 - epsilon) * autocorrelation_matrix(xj)
    return 0.5 * (Q + Q.conj().T)


def cross_correlation(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """脉压输出 ``r = H x``，长度 2N-1，``r[N-1] = h^H x``。

    等价于 ``build_convolution_matrix(h, N) @ x``，但不构造大矩阵。
    """
    x = np.asarray(x, dtype=complex)
    h = np.asarray(h, dtype=complex)
    # np.correlate(h, x)[k] = sum_n h[n+d] conj(x[n]) = (X h)[k]，而 H x = conj(X h)
    return np.conjugate(np.correlate(h, x, mode="full"))
