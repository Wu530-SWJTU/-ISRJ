"""PRD_01 §6 验收标准。"""

import numpy as np
import pytest

from isrj_joint_design.params import (
    GlobalParams,
    generate_isrj_jamming_signal,
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


def test_sampling_function_default_window(params):
    """默认窗口 [2.5, 5) μs（用户 2026-08-25 实验设定）：长度 N，元素 0/1，
    每周期第 50~99 个采样点为 1，占空比 0.25。"""
    j = generate_isrj_sampling_function(params)
    assert j.shape == (params.N,)
    assert set(np.unique(j)) <= {0.0, 1.0}
    assert j.mean() == pytest.approx(params.tau / params.T_s)
    assert params.N_tau_offset == 50
    for start in range(0, params.N, params.N_Ts):
        block = j[start:start + params.N_Ts]
        off = params.N_tau_offset
        assert np.all(block[:off] == 0.0)
        assert np.all(block[off:off + params.N_tau] == 1.0)
        assert np.all(block[off + params.N_tau:] == 0.0)


def test_sampling_function_prd_baseline_window():
    """PRD_01 §6 验收 1、2（基线行为）：tau_offset=0 时每周期前 N_tau 个为 1。"""
    p = GlobalParams(seed=0, tau_offset=0.0)
    j = generate_isrj_sampling_function(p)
    assert j.mean() == pytest.approx(p.tau / p.T_s)
    for start in range(0, p.N, p.N_Ts):
        block = j[start:start + p.N_Ts]
        assert np.all(block[:p.N_tau] == 1.0)
        assert np.all(block[p.N_tau:] == 0.0)


def test_sampling_window_offset():
    """采样窗口起点旋钮：tau_offset=0.25μs、tau=0.25μs → 每周期第 5~9 点为 1。

    默认参数（tau_offset=0）的 PRD 行为不受影响，见上一个测试。
    """
    p = GlobalParams(seed=0, tau=0.25e-6, tau_offset=0.25e-6)
    j = generate_isrj_sampling_function(p)
    assert p.N_tau == 5 and p.N_tau_offset == 5
    assert j.mean() == pytest.approx(p.tau / p.T_s)          # 占空比 0.025
    for start in range(0, p.N, p.N_Ts):
        block = j[start:start + p.N_Ts]
        assert np.all(block[:5] == 0.0)                      # [0, 0.25μs) 关
        assert np.all(block[5:10] == 1.0)                    # [0.25, 0.5)μs 开
        assert np.all(block[10:] == 0.0)                     # [0.5, 10)μs 关


def test_multi_repeat_jamming_signal():
    """多转发干扰生成（评估侧）：转发#1 截取完成后发（平移 50 点）、转发#2 平移 200 点。"""
    p = GlobalParams(seed=0, T_s=20e-6, tau_offset=0.0,
                     retransmit_offsets=(2.5e-6, 10e-6))
    rng = np.random.default_rng(1)
    x = np.exp(2j * np.pi * rng.random(p.N))
    j = generate_isrj_sampling_function(p)
    xj = x * j
    y = generate_isrj_jamming_signal(x, p)

    assert y.shape == (p.N + 200,)
    assert np.allclose(y[50:50 + p.N][j == 1], x[j == 1])          # 转发#1 平移 50
    assert np.allclose(y[200:200 + p.N][j == 1], x[j == 1])        # 转发#2 平移 200
    assert np.sum(np.abs(y) ** 2) == pytest.approx(2 * np.sum(np.abs(xj) ** 2))


def test_multi_repeat_jamming_single_copy_degenerates_to_sampled_waveform():
    """retransmit_offsets=(0,) 时退化为设计阶段的单份零延迟模型 x ⊙ j。"""
    p = GlobalParams(seed=0, T_s=20e-6, tau_offset=0.0, retransmit_offsets=(0.0,))
    rng = np.random.default_rng(2)
    x = np.exp(2j * np.pi * rng.random(p.N))
    y = generate_isrj_jamming_signal(x, p)
    assert y.shape == (p.N,)
    assert np.allclose(y, x * generate_isrj_sampling_function(p))


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
