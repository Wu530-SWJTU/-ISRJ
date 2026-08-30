"""PRD_02 §7 验收标准。"""

import numpy as np
import pytest

from isrj_joint_design.matrices import (
    assemble_P,
    assemble_Q,
    build_D_matrix,
    build_P,
    build_Q,
    build_convolution_matrix,
    build_correlation_matrix,
    build_isrj_convolution_matrix,
    build_isrj_correlation_matrix,
    cross_correlation,
)

N = 16
EPS = 0.5


@pytest.fixture
def case():
    rng = np.random.default_rng(2020)
    x = np.exp(2j * np.pi * rng.random(N))
    h = rng.normal(size=N) + 1j * rng.normal(size=N)
    j = (np.mod(np.arange(N), 4) < 1).astype(float)
    return x, h, j


def test_H_and_X_produce_the_same_correlation_output(case):
    """验收 1：Hx 与 Xh 描述同一输出。

    本实现的矩阵由共轭序列铺成（见 matrices 模块 docstring），因此二者互为
    共轭 —— 模长完全相同，目标函数因此严格相等。
    """
    x, h, j = case
    H = build_convolution_matrix(h, N)
    X = build_correlation_matrix(x, N)
    assert (H @ x).shape == (2 * N - 1,)
    assert np.allclose(H @ x, np.conjugate(X @ h), atol=1e-12)
    assert np.allclose(np.abs(H @ x), np.abs(X @ h), atol=1e-12)


def test_mainlobe_element_equals_the_constrained_inner_product(case):
    """验收 2：(Hx)[N] = h^H x，且 (Xh)[N] = x^H h。"""
    x, h, j = case
    H = build_convolution_matrix(h, N)
    X = build_correlation_matrix(x, N)
    assert (H @ x)[N - 1] == pytest.approx(np.vdot(h, x))
    assert (X @ h)[N - 1] == pytest.approx(np.vdot(x, h))


def test_isrj_matrices_are_consistent(case):
    """验收 3、4：Hj x = H(x⊙j)，且 Xj h 与 Hj x 一致（共轭意义下）。"""
    x, h, j = case
    H = build_convolution_matrix(h, N)
    Hj = build_isrj_convolution_matrix(h, j, N)
    Xj = build_isrj_correlation_matrix(x, j, N)
    assert np.allclose(Hj @ x, H @ (x * j), atol=1e-12)
    assert np.allclose(Hj @ x, np.conjugate(Xj @ h), atol=1e-12)
    assert (Hj @ x)[N - 1] == pytest.approx(np.vdot(h * j, x))


def test_D_matrix_masks_only_the_mainlobe():
    """验收 5：D 仅第 N 个对角元为 0。"""
    D = build_D_matrix(N)
    assert D.shape == (2 * N - 1, 2 * N - 1)
    assert np.array_equal(D, np.diag(np.diag(D)))
    assert D[N - 1, N - 1] == 0
    assert np.sum(np.diag(D)) == 2 * N - 2


def test_P_and_Q_are_hermitian(case):
    """验收 6：P、Q 为 Hermitian。"""
    x, h, j = case
    D = build_D_matrix(N)
    P = build_P(build_convolution_matrix(h, N), build_isrj_convolution_matrix(h, j, N), D, EPS)
    Q = build_Q(build_correlation_matrix(x, N), build_isrj_correlation_matrix(x, j, N), D, EPS)
    assert np.allclose(P, P.conj().T, atol=1e-12)
    assert np.allclose(Q, Q.conj().T, atol=1e-12)


def test_objective_is_the_same_from_both_viewpoints(case):
    """验收 7：x^H P x = h^H Q h。"""
    x, h, j = case
    D = build_D_matrix(N)
    P = build_P(build_convolution_matrix(h, N), build_isrj_convolution_matrix(h, j, N), D, EPS)
    Q = build_Q(build_correlation_matrix(x, N), build_isrj_correlation_matrix(x, j, N), D, EPS)
    obj_x = np.vdot(x, P @ x)
    obj_h = np.vdot(h, Q @ h)
    assert obj_x.imag == pytest.approx(0, abs=1e-9)
    assert obj_x.real == pytest.approx(obj_h.real, rel=1e-10)


def test_isl_term_removes_the_mainlobe(case):
    """验收 8：x^H (H^H D H) x = ||Hx||^2 - |(Hx)[N]|^2。"""
    x, h, j = case
    H = build_convolution_matrix(h, N)
    D = build_D_matrix(N)
    r = H @ x
    lhs = np.vdot(x, (H.conj().T @ D @ H) @ x).real
    assert lhs == pytest.approx(np.sum(np.abs(r) ** 2) - abs(r[N - 1]) ** 2, rel=1e-10)


def test_fast_assembly_matches_explicit_construction(case):
    """PRD_02 §4.3 的 Toeplitz 快速路径与显式构造等价。"""
    x, h, j = case
    D = build_D_matrix(N)
    P_ref = build_P(build_convolution_matrix(h, N), build_isrj_convolution_matrix(h, j, N), D, EPS)
    Q_ref = build_Q(build_correlation_matrix(x, N), build_isrj_correlation_matrix(x, j, N), D, EPS)
    assert np.allclose(assemble_P(h, j, EPS), P_ref, atol=1e-9)
    assert np.allclose(assemble_Q(x, j, EPS), Q_ref, atol=1e-9)


def test_cross_correlation_helper_matches_matrix_product(case):
    x, h, j = case
    assert np.allclose(cross_correlation(x, h), build_convolution_matrix(h, N) @ x, atol=1e-12)
    assert np.allclose(cross_correlation(x * j, h),
                       build_isrj_convolution_matrix(h, j, N) @ x, atol=1e-12)
