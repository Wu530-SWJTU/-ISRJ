"""模块一：全局参数初始化与 ISRJ 信号生成（PRD_01）。

所有下游模块的参数均从 :class:`GlobalParams` 读取，禁止在别处硬编码。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# 论文第二个仿真的 PAPR 扫描范围（PRD_01 §2.4）
PAPR_RANGE = (1.0, 1.05, 1.1, 1.15, 1.21, 1.44)


@dataclass
class GlobalParams:
    """论文 Table I / Section IV 的全部仿真参数。"""

    # --- 雷达信号参数 ---
    T_p: float = 40e-6          # 脉冲持续时间 [s]
    B: float = 20e6             # 带宽 [Hz]
    f_c: float = 1e9            # 载频 [Hz]
    N: int = 800                # 码长 = B * T_p（复采样）
    fs: float = 20e6            # 采样率 [Hz]，复采样默认等于带宽

    # --- 干扰机参数 ---
    T_s: float = 10e-6          # 采样重复周期 [s]
    tau: float = 2.5e-6         # 中断采样脉宽 [s]
    # 以下参数仅论文图 4 场景使用（干扰机时延/重发/SJR/几何位置），
    # 联合设计阶段（图 1-3）不建模，按 PRD_05 §8.3 保留但不参与计算。
    tau_delay_jammer: float = 2e-6
    num_repeats: int = 2
    SJR_db: float = -10.0
    pos_radar: tuple = (0.0, 0.0, 0.0)
    pos_jammer: tuple = (0.0, 2000.0, 0.0)
    pos_target: tuple = (0.0, 3000.0, 0.0)

    # --- 算法超参数 ---
    M: int = 100                # 外层交替优化轮数
    delta: float = 1e-4         # ADMM 收敛阈值 ||x-y||+||x-z||
    rho: float = 5000.0         # ADMM 惩罚参数
    epsilon: float = 0.5        # Pareto 权重
    max_inner_iter: int = 500   # ADMM 内层最大迭代次数

    # --- 约束参数 ---
    eta: float = 1.1            # PAPR 约束
    peak_interf_db: float = -30.0

    # --- 实现选项（非论文参数）---
    seed: int | None = None
    # PRD_01 §4.4 要求对偶变量用随机相位初始化；PRD_05 §2.1 的伪代码写为零向量。
    # 论文原文明确"dual variables are initialized with random-phase sequences"，故默认随机相位。
    dual_init: str = "random_phase"   # "random_phase" | "zeros"
    # 波形初始化（PRD_01 §4.1 指定随机相位，故为默认；论文仅要求随机相位序列，
    # 但实测 LFM 初始化的处理增益损失更接近论文报告的 -0.9dB，见 CLAUDE.md）。
    init: str = "random_phase"        # "random_phase" | "lfm"

    b_max: float = field(init=False)
    b_min: float = field(init=False)

    def __post_init__(self) -> None:
        self.b_max = float(self.N)
        # 20*log10(b_min/b_max) = peak_interf_db  ->  b_min = N * 10^(-30/20)
        self.b_min = self.b_max * 10 ** (self.peak_interf_db / 20.0)

    @property
    def N_Ts(self) -> int:
        """一个采样重复周期内的采样点数。"""
        return int(round(self.T_s * self.fs))

    @property
    def N_tau(self) -> int:
        """一个中断采样脉冲内的采样点数。"""
        return int(round(self.tau * self.fs))


def generate_isrj_sampling_function(params: GlobalParams) -> np.ndarray:
    """生成 ISRJ 离散采样函数 ``j``（PRD_01 §3）。

    每个周期的前 ``N_tau`` 个采样点为 1（干扰机采样），其余为 0。

    返回 shape=(N,) 的 float 数组，元素为 0.0 或 1.0。
    """
    n = np.arange(params.N)
    return (np.mod(n, params.N_Ts) < params.N_tau).astype(float)


def random_phase_vector(size: int, rng: np.random.Generator) -> np.ndarray:
    """单位模随机相位序列 ``exp(j*2*pi*U(0,1))``。"""
    return np.exp(2j * np.pi * rng.random(size))


def lfm_waveform(N: int, T_p: float, B: float) -> np.ndarray:
    """线性调频波形 ``exp(j*pi*B*t^2/T_p)``，单位模，用于可选的初始化。

    采样率取 ``fs = B``（PRD_01），故 ``t = n*T_p/N``。
    """
    t = np.arange(N) * (T_p / N)
    return np.exp(1j * np.pi * B * t ** 2 / T_p)


def initialize_variables(params: GlobalParams,
                         rng: np.random.Generator | None = None) -> dict:
    """初始化 x, h, y, z 及对偶变量 lambda, gamma, mu, nu（PRD_01 §4）。

    返回的字典即 ADMM 的热启动状态，外层迭代之间持续复用。
    """
    if rng is None:
        rng = np.random.default_rng(params.seed)

    N = params.N
    if params.init == "random_phase":
        x = random_phase_vector(N, rng)
    elif params.init == "lfm":
        x = lfm_waveform(N, params.T_p, params.B)
    else:
        raise ValueError(f"unknown init: {params.init!r}")
    h = random_phase_vector(N, rng)

    if params.dual_init == "zeros":
        lam = np.zeros(N, dtype=complex)
        gam = np.zeros(N, dtype=complex)
        mu = 0j
        nu = 0j
    elif params.dual_init == "random_phase":
        lam = random_phase_vector(N, rng)
        gam = random_phase_vector(N, rng)
        mu = complex(random_phase_vector(1, rng)[0])
        nu = complex(random_phase_vector(1, rng)[0])
    else:
        raise ValueError(f"unknown dual_init: {params.dual_init!r}")

    return {
        "x": x,
        "h": h,
        "y": x.copy(),      # y^H y = N 已满足（单位模）
        "z": x.copy(),      # |z[n]|^2 = 1 <= eta 已满足
        "lambda": lam,
        "gamma": gam,
        "mu": mu,
        "nu": nu,
    }
