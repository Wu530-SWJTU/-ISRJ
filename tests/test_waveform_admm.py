"""PRD_04 §8 验收标准。"""

import numpy as np
import pytest

from isrj_joint_design.filter_solver import solve_filter_lagrange
from isrj_joint_design.matrices import assemble_P, assemble_Q
from isrj_joint_design.params import GlobalParams, initialize_variables
from isrj_joint_design.waveform_admm import solve_waveform_admm

N = 64


def make_case(**overrides):
    """构造一次真实的 ADMM 调用现场。

    必须先走 Algorithm 1 的步骤 (a)(b) 解出 h 再调用 ADMM：随机单位模的初始 h
    会让约束集为空（``|h^H x| <= ||h|| ||x|| = N`` 取等要求 x=h，此时
    ``h_j^H x`` 等于 j 的非零个数，一般不等于 b_min），ADMM 无解可收敛。
    """
    params = GlobalParams(N=N, seed=5, **overrides)
    j = (np.mod(np.arange(N), 8) < 2).astype(float)
    variables = initialize_variables(params)
    h = solve_filter_lagrange(variables["x"], j, assemble_Q(variables["x"], j, params.epsilon),
                              params.b_max, params.b_min)
    variables["h"] = h
    P = assemble_P(h, j, params.epsilon)
    return params, j, P, variables


def test_x_update_solves_the_closed_form_linear_system():
    """验收 1：x 更新满足 A x = b_vec（式 14），残差 < 1e-8。"""
    params, j, P, variables = make_case(max_inner_iter=1)
    h = variables["h"]
    h_j = h * j
    x, _, _ = solve_waveform_admm(h, j, P, params, variables)

    rho = params.rho
    A = (2 * P + 2 * rho * np.eye(N)
         + rho * np.outer(h, h.conj()) + rho * np.outer(h_j, h_j.conj()))
    b_vec = (rho * (variables["z"] - variables["lambda"])
             + rho * (variables["y"] - variables["gamma"])
             + rho * (params.b_max - variables["mu"]) * h
             + rho * (params.b_min - variables["nu"]) * h_j)
    assert np.linalg.norm(A @ x - b_vec) / np.linalg.norm(b_vec) < 1e-8


def test_auxiliary_variables_stay_feasible():
    """验收 2、3：y^H y = N；max |z[n]|^2 <= eta。"""
    params, j, P, variables = make_case(max_inner_iter=20)
    _, v, _ = solve_waveform_admm(variables["h"], j, P, params, variables)
    assert abs(np.vdot(v["y"], v["y"]).real - N) / N < 1e-6
    assert np.max(np.abs(v["z"]) ** 2) <= params.eta + 1e-6


def test_dual_updates_follow_equation_13d():
    """验收 4：对偶变量按式 (13d) 更新，无遗漏。"""
    params, j, P, variables = make_case(max_inner_iter=1)
    h = variables["h"]
    h_j = h * j
    x, v, _ = solve_waveform_admm(h, j, P, params, variables)
    assert np.allclose(v["lambda"], x - v["z"] + variables["lambda"])
    assert np.allclose(v["gamma"], x - v["y"] + variables["gamma"])
    assert v["mu"] == pytest.approx(np.vdot(h, x) - params.b_max + variables["mu"])
    assert v["nu"] == pytest.approx(np.vdot(h_j, x) - params.b_min + variables["nu"])


def test_input_variables_are_not_mutated():
    """热启动状态由调用方管理，求解器不得就地修改入参。"""
    params, j, P, variables = make_case(max_inner_iter=5)
    snapshot = {k: (v.copy() if isinstance(v, np.ndarray) else v)
                for k, v in variables.items()}
    solve_waveform_admm(variables["h"], j, P, params, variables)
    for k, v in snapshot.items():
        if isinstance(v, np.ndarray):
            assert np.array_equal(variables[k], v)
        else:
            assert variables[k] == v


def test_consensus_error_decreases_and_converges():
    """验收 5：consensus error（两个 2-范数之和）降到 delta 以下。"""
    params, j, P, variables = make_case()
    _, _, info = solve_waveform_admm(variables["h"], j, P, params, variables)
    assert info["converged"], f"consensus error stalled at {info['consensus_error']:.3e}"
    assert info["consensus_error"] < params.delta
    assert info["iterations"] <= params.max_inner_iter


def test_objective_trends_downward():
    """验收 6：x^H P x 总体下降。"""
    params, j, P, variables = make_case()
    _, _, info = solve_waveform_admm(variables["h"], j, P, params, variables,
                                     record_objective=True)
    obj = info["objective"]
    assert len(obj) > 10
    assert obj[-1] < obj[0]
    assert np.mean(obj[-5:]) <= np.mean(obj[:5])


def test_output_waveform_satisfies_papr_constraint():
    """验收 7：PAPR(x) <= eta（允许微小数值误差）。"""
    params, j, P, variables = make_case()
    x, _, _ = solve_waveform_admm(variables["h"], j, P, params, variables)
    papr = np.max(np.abs(x) ** 2) / (np.sum(np.abs(x) ** 2) / N)
    assert papr <= params.eta + 1e-2


@pytest.mark.parametrize("rho", [1000.0, 5000.0, 20000.0])
def test_step_size_sensitivity(rho):
    """验收 8：不同 rho 下均能稳定运行（rho=5000 为论文取值）。"""
    params, j, P, variables = make_case(rho=rho)
    x, _, info = solve_waveform_admm(variables["h"], j, P, params, variables)
    assert np.all(np.isfinite(x))
    assert info["consensus_error"] < 1e-2
