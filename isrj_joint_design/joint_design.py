"""模块五（上）：交替优化主算法与性能指标（PRD_05，论文 Algorithm 1）。"""

from __future__ import annotations

import time

import numpy as np

from .filter_solver import solve_filter_lagrange
from .matrices import assemble_P, assemble_Q, cross_correlation
from .params import GlobalParams, generate_isrj_sampling_function, initialize_variables
from .waveform_admm import solve_waveform_admm


# --------------------------------------------------------------------------
# 性能指标
# --------------------------------------------------------------------------

def compute_metrics(x: np.ndarray, h: np.ndarray, j: np.ndarray) -> dict:
    """计算论文关心的全部标量指标。

    - ``isl``  : 波形滤波器输出的积分旁瓣电平 ``||r||^2 - |r[N]|^2``
    - ``il``   : ISRJ 信号滤波器输出的积分电平 ``||r_j||^2``
    - ``psl_db``: 峰值旁瓣电平（相对主瓣，dB）
    - ``isrj_zero_delay_db``: ISRJ 输出在零延迟处的电平，由 b_min 约束**钉死**在 -30dB
    - ``isrj_peak_db``: ISRJ 输出在全部延迟上的最大值，由 ILs 目标项压低（无硬约束）
    - ``processing_gain_loss_db``: ``10 log10(|h^H x|^2 / (||h||^2 ||x||^2))``
    - ``papr`` : ``max|x[n]|^2 / (||x||^2 / N)``
    """
    x = np.asarray(x, dtype=complex)
    h = np.asarray(h, dtype=complex)
    N = x.size

    r = cross_correlation(x, h)
    r_j = cross_correlation(x * np.asarray(j, dtype=float), h)

    peak = abs(r[N - 1])
    sidelobes = np.delete(np.abs(r), N - 1)

    return {
        "isl": float(np.sum(np.abs(r) ** 2) - peak ** 2),
        "il": float(np.sum(np.abs(r_j) ** 2)),
        "peak": float(peak),
        "psl_db": float(20 * np.log10(sidelobes.max() / peak)),
        "isrj_zero_delay_db": float(20 * np.log10(abs(r_j[N - 1]) / peak)),
        "isrj_peak_db": float(20 * np.log10(np.abs(r_j).max() / peak)),
        "processing_gain_loss_db": float(
            10 * np.log10(abs(np.vdot(h, x)) ** 2
                          / (np.sum(np.abs(h) ** 2) * np.sum(np.abs(x) ** 2)))
        ),
        "papr": float(np.max(np.abs(x) ** 2) / (np.sum(np.abs(x) ** 2) / N)),
    }


# --------------------------------------------------------------------------
# Algorithm 1
# --------------------------------------------------------------------------

def joint_design_algorithm(params: GlobalParams,
                           j: np.ndarray | None = None,
                           verbose: bool = True,
                           log_every: int = 10) -> dict:
    """交替优化主算法：外层更新 h（解析解）与 x（ADMM），共 ``params.M`` 轮。

    对偶变量与辅助变量在外层迭代之间**热启动**（不重新初始化），这是 ADMM
    收敛的关键（PRD_05 §2.2）。
    """
    if j is None:
        j = generate_isrj_sampling_function(params)
    j = np.asarray(j, dtype=float)

    variables = initialize_variables(params)
    x = variables["x"]
    h = variables["h"]

    isl_history: list[float] = []
    il_history: list[float] = []
    loss_history: list[float] = []
    admm_iters: list[int] = []
    consensus_errors: list[float] = []

    t0 = time.perf_counter()
    for m in range(1, params.M + 1):
        # (a)(b) 固定 x，构造 Q，解析求解滤波器
        Q = assemble_Q(x, j, params.epsilon)
        h = solve_filter_lagrange(x, j, Q, params.b_max, params.b_min)

        # (c)(d) 固定 h，构造 P，ADMM 求解波形
        P = assemble_P(h, j, params.epsilon)
        variables["x"] = x
        x, variables, info = solve_waveform_admm(h, j, P, params, variables)

        # (e) 记录目标函数
        metrics = compute_metrics(x, h, j)
        isl_history.append(metrics["isl"])
        il_history.append(metrics["il"])
        loss_history.append(metrics["processing_gain_loss_db"])
        admm_iters.append(info["iterations"])
        consensus_errors.append(info["consensus_error"])

        if verbose and (m % log_every == 0 or m == 1 or m == params.M):
            print(f"  [{m:3d}/{params.M}] ISLs={metrics['isl']:.4e} "
                  f"ILs={metrics['il']:.4e} "
                  f"loss={metrics['processing_gain_loss_db']:6.2f}dB "
                  f"admm_iter={info['iterations']:3d} "
                  f"cons_err={info['consensus_error']:.3e}", flush=True)

    elapsed = time.perf_counter() - t0
    metrics = compute_metrics(x, h, j)
    xj = x * j
    result = {
        "x": x,
        "h": h,
        "j": j,
        "isl_history": isl_history,
        "il_history": il_history,
        "loss_history": loss_history,
        "admm_iterations": admm_iters,
        "consensus_errors": consensus_errors,
        "final_isl": isl_history[-1],
        "final_il": il_history[-1],
        "elapsed_s": elapsed,
        # 约束残差（PRD_05 §6.1 验收 2）
        "residual_b_max": float(abs(np.vdot(x, h) - params.b_max) / params.b_max),
        "residual_b_min": float(abs(np.vdot(xj, h) - params.b_min) / params.b_min),
        **metrics,
    }
    result["processing_gain_loss"] = result["processing_gain_loss_db"]
    return result


def summarize(result: dict, params: GlobalParams) -> str:
    """把一次联合设计的关键指标格式化成一段可读文本。"""
    return "\n".join([
        f"耗时                 : {result['elapsed_s']:.1f} s "
        f"(M={params.M}, N={params.N}, eta={params.eta})",
        f"最终 ISLs            : {result['final_isl']:.4e}",
        f"最终 ILs             : {result['final_il']:.4e}",
        f"处理增益损失         : {result['processing_gain_loss_db']:.3f} dB (论文 -0.9 dB)",
        f"PSL                  : {result['psl_db']:.2f} dB",
        f"ISRJ 零延迟电平      : {result['isrj_zero_delay_db']:.2f} dB (约束 = -30 dB)",
        f"ISRJ 全延迟峰值      : {result['isrj_peak_db']:.2f} dB (由 ILs 目标项压低)",
        f"PAPR                 : {result['papr']:.4f} (约束 <= {params.eta})",
        f"主瓣约束相对残差     : {result['residual_b_max']:.2e}",
        f"干扰峰值约束相对残差 : {result['residual_b_min']:.2e}",
        f"末次 ADMM consensus  : {result['consensus_errors'][-1]:.2e} (阈值 {params.delta})",
    ])
