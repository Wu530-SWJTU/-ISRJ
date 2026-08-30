"""PRD_01 §6 验收标准。"""

import numpy as np
import pytest

from isrj_joint_design.params import (
    GlobalParams,
    generate_isrj_sampling_function,
    initialize_variables,
)


@pytest.fixture
def params():
    return GlobalParams(seed=0)


def test_derived_constraint_values(params):
    assert params.N == 800
    assert params.b_max == 800
    assert params.b_min == pytest.approx(800 * 10 ** -1.5)
    assert 20 * np.log10(params.b_min / params.b_max) == pytest.approx(-30.0)
    assert (params.N_Ts, params.N_tau) == (200, 50)


def test_sampling_function_is_binary_with_correct_duty_cycle(params):
    """验收 1、2：长度 N，元素 0/1，每周期前 N_tau 个为 1，占空比 0.25。"""
    j = generate_isrj_sampling_function(params)
    assert j.shape == (params.N,)
    assert set(np.unique(j)) <= {0.0, 1.0}
    assert j.mean() == pytest.approx(params.tau / params.T_s)
    for start in range(0, params.N, params.N_Ts):
        block = j[start:start + params.N_Ts]
        assert np.all(block[:params.N_tau] == 1.0)
        assert np.all(block[params.N_tau:] == 0.0)


def test_initial_waveform_and_filter_are_unit_modulus(params):
    """验收 3：x、h 为单位模随机相位向量，||x||^2 = ||h||^2 = N。"""
    v = initialize_variables(params)
    for key in ("x", "h"):
        assert np.allclose(np.abs(v[key]), 1.0)
        assert np.sum(np.abs(v[key]) ** 2) == pytest.approx(params.N)


def test_auxiliary_variables_satisfy_their_constraints(params):
    """验收 4：y^H y = N，|z[n]|^2 <= eta。"""
    v = initialize_variables(params)
    assert np.vdot(v["y"], v["y"]).real == pytest.approx(params.N)
    assert np.max(np.abs(v["z"]) ** 2) <= params.eta


def test_dual_variables_are_random_phase_by_default(params):
    """验收 5：对偶变量默认按论文用随机相位单位模初始化。"""
    v = initialize_variables(params)
    assert np.allclose(np.abs(v["lambda"]), 1.0)
    assert np.allclose(np.abs(v["gamma"]), 1.0)
    assert abs(v["mu"]) == pytest.approx(1.0)
    assert abs(v["nu"]) == pytest.approx(1.0)


def test_dual_variables_can_be_zeroed():
    """PRD_05 §2.1 伪代码的零初始化作为可选项保留。"""
    v = initialize_variables(GlobalParams(seed=0, dual_init="zeros"))
    assert np.all(v["lambda"] == 0) and np.all(v["gamma"] == 0)
    assert v["mu"] == 0 and v["nu"] == 0


def test_seed_makes_initialization_reproducible():
    a = initialize_variables(GlobalParams(seed=7))
    b = initialize_variables(GlobalParams(seed=7))
    assert np.array_equal(a["x"], b["x"]) and np.array_equal(a["h"], b["h"])


def test_lfm_initialization_option():
    """可选的 LFM 波形初始化：单位模、确定性、满足全部初始约束。"""
    v = initialize_variables(GlobalParams(seed=0, init="lfm"))
    assert np.allclose(np.abs(v["x"]), 1.0)
    assert np.sum(np.abs(v["x"]) ** 2) == pytest.approx(800)
    assert not np.allclose(v["x"], np.ones(800))          # 确为调频而非恒模常数
    # 与同 seed 的随机相位初始化不同，且与 seed 无关（确定性）
    w = initialize_variables(GlobalParams(seed=1, init="lfm"))
    assert np.array_equal(v["x"], w["x"])
