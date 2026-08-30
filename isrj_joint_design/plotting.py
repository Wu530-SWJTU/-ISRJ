"""模块五（下）：论文图 1–图 3 的绘图（PRD_05 §3–§4）。

配色取自 dataviz 参考调色板的分类色槽 1/2（蓝 #2a78d6 / 橙 #eb6834），两条
曲线时始终带图例，单条曲线不加图例（标题已指明）；网格弱化，文字用墨色而
非series色，全部单一纵轴。
"""

from __future__ import annotations

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .joint_design import compute_metrics  # noqa: E402
from .matrices import cross_correlation  # noqa: E402
from .params import GlobalParams  # noqa: E402

SERIES_1 = "#2a78d6"    # blue
SERIES_2 = "#eb6834"    # orange
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
SURFACE = "#fcfcfb"
GRID = "#d8d7d2"

_FLOOR = 1e-30          # 防止 log10(0)


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    ax.xaxis.label.set_color(TEXT_SECONDARY)
    ax.yaxis.label.set_color(TEXT_SECONDARY)
    ax.title.set_color(TEXT_PRIMARY)


def _save(fig, save_path: str | None):
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, facecolor=SURFACE)
        print(f"  saved: {save_path}")
    plt.close(fig)


def _to_db(values: np.ndarray, ref: float) -> np.ndarray:
    return 20 * np.log10(np.maximum(np.abs(values), _FLOOR) / ref)


# --------------------------------------------------------------------------
# 图 2：相关性图像（核心需求，PRD_05 §3）
# --------------------------------------------------------------------------

def plot_correlation_comparison(x: np.ndarray, h: np.ndarray, j: np.ndarray,
                                params: GlobalParams,
                                save_path: str | None = None) -> None:
    """图 2(a)(b)：无 ISRJ / 有 ISRJ 干扰下的相关性（共用归一化因子）。"""
    N = params.N
    r = cross_correlation(x, h)
    r_j = cross_correlation(np.asarray(x) * np.asarray(j, dtype=float), h)
    ref = abs(r[N - 1])                       # 两图共用的归一化因子：波形主瓣峰值

    k = np.arange(2 * N - 1)
    r_db, r_j_db = _to_db(r, ref), _to_db(r_j, ref)
    m = compute_metrics(x, h, j)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), facecolor=SURFACE)

    ax = axes[0]
    ax.plot(k, r_db, color=SERIES_1, linewidth=1.2)
    ax.set_title("(a) Waveform vs. mismatched filter", fontsize=11)
    ax.set_xlabel("Delay (samples)")
    ax.set_ylabel("Normalized correlation (dB)")
    ax.set_xlim(0, 2 * N - 2)
    ax.set_ylim(-60, 5)
    ax.annotate(f"PSL = {m['psl_db']:.1f} dB\nprocessing gain loss = "
                f"{m['processing_gain_loss_db']:.2f} dB",
                xy=(0.03, 0.06), xycoords="axes fraction",
                fontsize=9, color=TEXT_PRIMARY)
    _style(ax)

    ax = axes[1]
    ax.plot(k, r_j_db, color=SERIES_2, linewidth=1.2)
    lo = max(-400, np.floor(r_j_db[np.isfinite(r_j_db)].min() / 50) * 50)
    ax.set_title("(b) ISRJ signal vs. mismatched filter", fontsize=11)
    ax.set_xlabel("Delay (samples)")
    ax.set_ylabel("Normalized correlation (dB)")
    ax.set_xlim(0, 2 * N - 2)
    ax.set_ylim(lo, 5)
    ax.axhline(-30, color=TEXT_SECONDARY, linewidth=1.0, linestyle="--")
    ax.annotate(f"ISRJ peak = {m['isrj_peak_db']:.1f} dB  (limit -30 dB)",
                xy=(0.03, 0.06), xycoords="axes fraction",
                fontsize=9, color=TEXT_PRIMARY)
    _style(ax)

    _save(fig, save_path)


# --------------------------------------------------------------------------
# 图 1：收敛曲线与波形
# --------------------------------------------------------------------------

def plot_convergence(isl_history: list, il_history: list,
                     save_path: str | None = None) -> None:
    """图 1(a)：ISLs / ILs 随外层迭代次数的收敛曲线（对数纵轴，单一纵轴）。"""
    it = np.arange(1, len(isl_history) + 1)
    fig, ax = plt.subplots(figsize=(6.5, 4.5), facecolor=SURFACE)
    ax.semilogy(it, isl_history, color=SERIES_1, linewidth=2,
                label="ISLs of waveform filter output")
    ax.semilogy(it, il_history, color=SERIES_2, linewidth=2,
                label="ILs of ISRJ signal filter output")
    for hist, color in ((isl_history, SERIES_1), (il_history, SERIES_2)):
        ax.plot(it[-1], hist[-1], "o", color=color, markersize=8)
    ax.annotate(f"{isl_history[-1]:.2e}", xy=(it[-1], isl_history[-1]),
                xytext=(-6, 10), textcoords="offset points",
                ha="right", fontsize=9, color=TEXT_PRIMARY)
    ax.annotate(f"{il_history[-1]:.2e}", xy=(it[-1], il_history[-1]),
                xytext=(-6, -16), textcoords="offset points",
                ha="right", fontsize=9, color=TEXT_PRIMARY)
    ax.set_title("Convergence of the objective terms", fontsize=11)
    ax.set_xlabel("Outer iteration")
    ax.set_ylabel("Integrated level")
    ax.legend(frameon=False, fontsize=9, labelcolor=TEXT_SECONDARY)
    _style(ax)
    _save(fig, save_path)


def plot_waveform(x: np.ndarray, params: GlobalParams,
                  save_path: str | None = None) -> None:
    """图 1(b)：设计波形的实部，并标注 PAPR。"""
    x = np.asarray(x)
    papr = float(np.max(np.abs(x) ** 2) / (np.sum(np.abs(x) ** 2) / x.size))
    fig, ax = plt.subplots(figsize=(7.5, 4.0), facecolor=SURFACE)
    ax.plot(np.arange(x.size), x.real, color=SERIES_1, linewidth=1.0)
    ax.set_title("Real part of the designed waveform", fontsize=11)
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Real part of x(t)")
    ax.set_xlim(0, x.size)
    ax.annotate(f"PAPR = {papr:.3f}  (constraint {params.eta})",
                xy=(0.02, 0.04), xycoords="axes fraction",
                fontsize=9, color=TEXT_PRIMARY)
    _style(ax)
    _save(fig, save_path)


# --------------------------------------------------------------------------
# 图 3：PAPR 扫描
# --------------------------------------------------------------------------

def plot_papr_sweep(results: dict, save_path: str | None = None) -> None:
    """图 3(a)(b)：处理增益损失、ISLs/ILs 随 PAPR 的变化。

    ``results`` 形如 ``{"papr": [...], "loss_db": [...], "isl": [...], "il": [...]}``。
    """
    papr = np.asarray(results["papr"], dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), facecolor=SURFACE)

    ax = axes[0]
    ax.plot(papr, results["loss_db"], color=SERIES_1, linewidth=2,
            marker="o", markersize=8)
    for p, v in zip(papr, results["loss_db"]):
        ax.annotate(f"{v:.2f}", xy=(p, v), xytext=(0, 9),
                    textcoords="offset points", ha="center",
                    fontsize=8, color=TEXT_PRIMARY)
    ax.set_title("(a) Loss of processing gain vs. PAPR", fontsize=11)
    ax.set_xlabel("PAPR")
    ax.set_ylabel("Loss of processing gain (dB)")
    _style(ax)

    ax = axes[1]
    ax.semilogy(papr, results["isl"], color=SERIES_1, linewidth=2,
                marker="o", markersize=8, label="ISLs")
    ax.semilogy(papr, results["il"], color=SERIES_2, linewidth=2,
                marker="s", markersize=8, label="ILs")
    ax.set_title("(b) ISLs and ILs vs. PAPR", fontsize=11)
    ax.set_xlabel("PAPR")
    ax.set_ylabel("Integrated level")
    ax.legend(frameon=False, fontsize=9, labelcolor=TEXT_SECONDARY)
    _style(ax)

    _save(fig, save_path)
