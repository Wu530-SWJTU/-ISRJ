"""雷达抗 ISRJ 波形—失配滤波器联合设计。

复现 K. Zhou, D. Li, Y. Su, T. Liu, "Joint Design of Transmit Waveform and
Mismatch Filter in the Presence of Interrupted Sampling Repeater Jamming,"
IEEE Signal Processing Letters, 2020. DOI: 10.1109/LSP.2020.3021667

模块划分与 PRD 文档一一对应：
    params        -> PRD_01 全局参数与 ISRJ 信号生成
    matrices      -> PRD_02 矩阵构造与目标函数
    filter_solver -> PRD_03 拉格朗日乘子法求解滤波器
    waveform_admm -> PRD_04 ADMM 交替优化求解波形
    joint_design  -> PRD_05 主算法与指标
    plotting      -> PRD_05 相关性绘图
"""

from .filter_solver import solve_filter_lagrange
from .joint_design import compute_metrics, joint_design_algorithm, summarize
from .matrices import assemble_P, assemble_Q, cross_correlation
from .params import GlobalParams, generate_isrj_sampling_function, initialize_variables
from .waveform_admm import solve_waveform_admm

__all__ = [
    "GlobalParams",
    "generate_isrj_sampling_function",
    "initialize_variables",
    "assemble_P",
    "assemble_Q",
    "cross_correlation",
    "solve_filter_lagrange",
    "solve_waveform_admm",
    "joint_design_algorithm",
    "compute_metrics",
    "summarize",
]
