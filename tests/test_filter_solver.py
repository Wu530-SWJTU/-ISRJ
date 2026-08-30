"""PRD_03 §6 验收标准。"""

import numpy as np
import pytest

from isrj_joint_design.filter_solver import filter_constraint_residuals, solve_filter_lagrange
from isrj_joint_design.matrices import assemble_Q

N = 32
EPS = 0.5
B_MAX = float(N)
B_MIN = float(N) * 10 ** -1.5


@pytest.fixture
def case():
    rng = np.random.default_rng(11)
    x = np.exp(2j * np.pi * rng.random(N))
    j = (np.mod(np.arange(N), 8) < 2).astype(float)
    Q = assemble_Q(x, j, EPS)
    return x, j, Q


def test_solution_satisfies_both_constraints(case):
    """验收 1：x^H h = b_max 且 x_j^H h = b_min（相对误差 < 1e-6）。"""
    x, j, Q = case
    h = solve_filter_lagrange(x, j, Q, B_MAX, B_MIN)
    r_max, r_min = filter_constraint_residuals(x, h, j, B_MAX, B_MIN)
    assert r_max < 1e-6
    assert r_min < 1e-6


def test_solution_is_optimal_against_feasible_perturbations(case):
    """验收 2：任意可行方向上目标函数都不下降（KKT 最优性）。"""
    x, j, Q = case
    xj = x * j
    h = solve_filter_lagrange(x, j, Q, B_MAX, B_MIN)
    base = np.vdot(h, Q @ h).real

    # 可行方向 d 张成 {x, x_j} 的正交补：x^H d = x_j^H d = 0
    basis = np.column_stack([x, xj])
    proj = basis @ np.linalg.pinv(basis.conj().T @ basis) @ basis.conj().T
    rng = np.random.default_rng(3)
    for scale in (1e-3, 1e-2, 1e-1, 1.0):
        v = rng.normal(size=N) + 1j * rng.normal(size=N)
        d = scale * (v - proj @ v)
        assert abs(np.vdot(x, d)) < 1e-9 and abs(np.vdot(xj, d)) < 1e-9
        assert np.vdot(h + d, Q @ (h + d)).real >= base - 1e-9 * max(base, 1.0)


def test_matches_kkt_linear_system_solution(case):
    """验收 2（补充）：与直接解 KKT 线性方程组的结果一致。"""
    x, j, Q = case
    xj = x * j
    A = np.column_stack([x, xj]).conj().T          # 约束矩阵，A h = [b_max, b_min]
    kkt = np.block([[2 * Q, A.conj().T], [A, np.zeros((2, 2), dtype=complex)]])
    rhs = np.concatenate([np.zeros(N, dtype=complex), [B_MAX, B_MIN]])
    h_ref = np.linalg.solve(kkt, rhs)[:N]
    h = solve_filter_lagrange(x, j, Q, B_MAX, B_MIN)
    assert np.allclose(h, h_ref, rtol=1e-6, atol=1e-8)


def test_numerically_stable_for_random_waveforms(case):
    """验收 4：随机 x 下无 NaN/Inf，且行列式 D = ac - |b|^2 > 0。"""
    _, j, _ = case
    for seed in range(5):
        rng = np.random.default_rng(seed)
        x = np.exp(2j * np.pi * rng.random(N))
        Q = assemble_Q(x, j, EPS)
        h = solve_filter_lagrange(x, j, Q, B_MAX, B_MIN)
        assert np.all(np.isfinite(h))
        r_max, r_min = filter_constraint_residuals(x, h, j, B_MAX, B_MIN)
        assert r_max < 1e-6 and r_min < 1e-6
