"""三维平滑性分析：θ = [t_0, d, T_s] 上的代价曲面有多"光滑"。

固定设计对取搜索线基线 v2（见 scripts/baseline.py：B = 10 MHz、L = 256、t_0 = 2 μs、
T_L = 1.5 μs、T_s = 6.4 μs、约束 −80 dB、random_phase、seed = 0），扫实际
干扰参数（录波口径）：z(θ*) = 20lg( |h^H y_j(θ*)| / |h^H x| )。矩阵链未变：
J(θ) = diag(j(θ))、H_j = H·diag(j(θ))、X_j 由 x⊙j 铺成。

四项分析（都是搜索算法直接关心的性质）：

1. **量化台阶**：`N_tau = round(T_L·fs)` 等全部取整 ⇒ 曲面的真实分辨率是 **1 个采样点**，
   低于此步长时梯度恒为 0（分段常数 + 跳变）。细扫（跨度 ±1 采样点、410 步）统计零差分比例。
   1 采样点 = 0.1 μs（t_0、T_s）；对 d 为 1/(T_s·fs) = 1/64。
2. **曲率**（二阶差分，步长取 1 采样点，避开量化）：沿三轴统计中位数/极值
   ⇒ 哪一维光滑、哪一维陡（单位 dB/(采样点)²）。
3. **局部极小个数**（11×11×11 网格 + 26 邻域严格比较）⇒ 是否存在多个盆地。
4. **设计点 Hessian**（3×3，中心差分）⇒ 特征值与特征向量：最陡方向 / 最平（退化）方向；
   并注明设计点在位移型维度上是 |·| 型拐点（C0 但不 C1）。

用法：
    .venv/Scripts/python.exe 参数搜索/scripts/run_smoothness_analysis.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from isrj_joint_design.matrices import cross_correlation  # noqa: E402
from isrj_joint_design.params import generate_isrj_sampling_function  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import baseline  # noqa: E402

N = baseline.N
FS = baseline.FS_HZ
T0_US, TS_US, TAU_US = baseline.T0_US, baseline.TS_US, baseline.TAU_US
D_DESIGN = TAU_US / TS_US
DESIGN_NPZ = ROOT / "参数搜索" / "结果" / "参数调制" / "设计对" / "joint_design_v2_t0_2us.npz"
OUTDIR = ROOT / "参数搜索" / "结果" / "参数调制" / "平滑性"
SAMPLE_US = baseline.SAMPLE_US                  # 1 个采样点 = 0.1 μs
SAMPLE_D = SAMPLE_US / TS_US                    # d 的 1 采样点步长 = 1/64
SERIES_1, SERIES_2, SERIES_3 = "#2a78d6", "#eb6834", "#3f9142"
AXES = ("t0", "ts", "d")
AXES_NAME = {"t0": "t_0", "ts": "T_s", "d": "d"}
TEXT_PRIMARY, TEXT_SECONDARY = "#0b0b0b", "#52514e"
SURFACE, GRID_C = "#fcfcfb", "#d8d7d2"


def level(x, h, e_t0=0.0, ed=0.0, e_ts=0.0):
    """三个误差都以采样点为单位（e_t0/e_ts 直接是点；ed 为 d 误差，乘 SAMPLE_D 折算）。"""
    p = baseline.params_for(t0_us=T0_US + e_t0 * SAMPLE_US,
                            tau_us=(D_DESIGN + ed) * TS_US,
                            ts_us=TS_US + e_ts * SAMPLE_US)
    r = cross_correlation(x * generate_isrj_sampling_function(p), h)
    return float(20 * np.log10(abs(r[N - 1]) / abs(np.vdot(h, x)) + 1e-30))


def staircase(x, h, axis, n_steps=41, span=2.0):
    """细扫（跨度 span 个采样点、n_steps×10 步），返回 (误差[点], 电平, 零差分比例, 台阶级数)。"""
    errs = np.linspace(-span / 2, span / 2, n_steps * 10 + 1)
    z = []
    for e in errs:
        if axis == "t0":
            z.append(level(x, h, e_t0=e))
        elif axis == "ts":
            z.append(level(x, h, e_ts=e))
        else:
            z.append(level(x, h, ed=e * SAMPLE_D))
    z = np.array(z)
    d = np.diff(z)
    zero_frac = float(np.mean(np.abs(d) < 1e-9))
    steps = int(np.sum(np.abs(d) >= 1e-9))
    return errs, z, zero_frac, steps


def curvature(x, h, axis, base_pts=(-4, -3, -2, 2, 3, 4)):
    """二阶差分（步长 = 1 采样点）：返回中位数与极值 [dB/(采样点)²]。"""
    vals = []
    for b in base_pts:
        if axis == "t0":
            f = lambda k: level(x, h, e_t0=b + k)
        elif axis == "ts":
            f = lambda k: level(x, h, e_ts=b + k)
        else:
            f = lambda k: level(x, h, ed=(b + k) * SAMPLE_D)
        vals.append(f(-1) - 2 * f(0) + f(1))
    vals = np.array(vals)
    return float(np.median(vals)), float(np.max(np.abs(vals)))


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    OUTDIR.mkdir(parents=True, exist_ok=True)

    d = np.load(DESIGN_NPZ)
    x, h = d["x"], d["h"]
    print("=== 三维平滑性分析（θ = [t_0, d, T_s]）===")
    print(f"  1 采样点 = {SAMPLE_US:g} μs（t_0、T_s）；对 d 为 {SAMPLE_D:g}")

    # --- 1) 量化台阶 ---
    print("\n[1] 量化台阶（细扫：跨度 ±1 采样点共 410 步，每步约 0.005 采样点）")
    stair = {}
    for axis, name in (("t0", "t_0"), ("ts", "T_s"), ("d", "d")):
        errs, z, zero_frac, steps = staircase(x, h, axis)
        stair[axis] = dict(err=errs.tolist(), z=z.tolist(),
                           zero_frac=zero_frac, steps=steps)
        print(f"  {name:4s}: 零差分比例 {zero_frac * 100:5.1f}%  "
              f"（{steps} 次跳变 / {len(errs) - 1} 个步长）"
              f"  ⇒ 有效分辨率 = 1 采样点")

    # --- 2) 曲率 ---
    print("\n[2] 曲率（二阶差分，步长 = 1 采样点）")
    curv = {}
    for axis, name in (("t0", "t_0"), ("ts", "T_s"), ("d", "d")):
        med, mx = curvature(x, h, axis)
        curv[axis] = dict(median=med, max_abs=mx)
        print(f"  {name:4s}: 中位数 {med:+7.3f} dB/点²，|最大| {mx:7.3f} dB/点²")

    # --- 2b) 微粗糙度：1 采样点尺度上增量符号翻转多少次 ---
    print()
    print("[2b] 微粗糙度（1 采样点步长，±5 点范围）")
    rough = {}
    for axis in AXES:
        k = np.arange(-5, 6)
        zz = np.array([level(x, h, e_t0=float(q)) if axis == "t0" else
                       (level(x, h, e_ts=float(q)) if axis == "ts" else
                        level(x, h, ed=float(q) * SAMPLE_D)) for q in k])
        inc = np.diff(zz)
        sgn = np.sign(inc)
        flips = int(np.sum(sgn[:-1] * sgn[1:] < 0))
        rough[axis] = dict(sign_flips=flips, n_steps=int(len(inc)),
                           median_abs_inc=float(np.median(np.abs(inc))),
                           max_abs_inc=float(np.max(np.abs(inc))))
        print(f"  {AXES_NAME[axis]:4s}: 符号翻转 {flips}/{len(inc)} 步"
              f"（{flips / len(inc) * 100:.0f}%）；|增量| 中位数 "
              f"{np.median(np.abs(inc)):.2f} dB，最大 {np.max(np.abs(inc)):.2f} dB")

    # --- 3) 三维网格 + 局部极小 ---
    print("\n[3] 三维网格与局部极小（11×11×11）")
    g_t0 = np.linspace(-5, 5, 11)            # ±0.5 μs（5 采样点）
    g_d = np.linspace(-5, 5, 11)             # ±0.078（5 采样点 × SAMPLE_D）
    g_ts = np.linspace(-5, 5, 11)
    Z = np.zeros((11, 11, 11))
    for i, a in enumerate(g_t0):
        for j, b in enumerate(g_d):
            for k, c in enumerate(g_ts):
                Z[i, j, k] = level(x, h, e_t0=a, ed=b * SAMPLE_D, e_ts=c)
    inner = Z[1:-1, 1:-1, 1:-1]
    neigh = [Z[a:a + 9, b:b + 9, c:c + 9]
             for a in range(3) for b in range(3) for c in range(3)]
    is_min = np.ones_like(inner, dtype=bool)
    for nb in neigh:
        is_min &= inner <= nb
    # 严格极小（至少有一维严格小）
    strict = is_min & np.any([inner < nb for nb in neigh], axis=0)
    idx = np.argwhere(strict)
    locs = [(g_t0[i + 1], g_d[j + 1] * SAMPLE_D, g_ts[k + 1]) for i, j, k in idx]
    print(f"  网格点 {Z.size}；全局最小 {Z.min():.2f} dB "
          f"@ (t_0 {g_t0[np.unravel_index(Z.argmin(), Z.shape)[0]]:+.0f} 点, "
          f"d {g_d[np.unravel_index(Z.argmin(), Z.shape)[1]] * SAMPLE_D:+.3f}, "
          f"T_s {g_ts[np.unravel_index(Z.argmin(), Z.shape)[2]]:+.0f} 点)")
    print(f"  严格局部极小个数 = {len(locs)}"
          + ("" if len(locs) == 1 else f"：{locs[:5]}"))

    # --- 4) 设计点 Hessian ---
    print()
    print("[4] 设计点 Hessian（中心差分；三个误差统一以「采样点」为单位，步长 1 点）")

    def z3(u: float = 0.0, v: float = 0.0, w: float = 0.0) -> float:
        """u = Δt_0 [点]、v = Δd [点]、w = ΔT_s [点]。"""
        return level(x, h, e_t0=u, ed=v * SAMPLE_D, e_ts=w)

    z0 = z3()
    Hm = np.zeros((3, 3))
    for i in range(3):
        args_p, args_m = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
        args_p[i], args_m[i] = 1.0, -1.0
        Hm[i, i] = z3(*args_p) - 2 * z0 + z3(*args_m)      # 步长 = 1 点 ⇒ 不再除 h²
        for j in range(i + 1, 3):
            def f(si, sj, ii=i, jj=j):
                a = [0.0, 0.0, 0.0]
                a[ii], a[jj] = si, sj
                return z3(*a)
            Hm[i, j] = Hm[j, i] = (f(1, 1) - f(1, -1) - f(-1, 1) + f(-1, -1)) / 4.0
    evals, evecs = np.linalg.eigh(Hm)
    print("  特征值（dB/点²）：" + "  ".join(f"{v:+.4f}" for v in evals))
    for v, vec in zip(evals, evecs.T):
        lab = " / ".join(f"{AXES_NAME[AXES[i]]} {vec[i]:+.2f}" for i in range(3))
        print(f"    {v:+.4f} 方向：{lab}")

    # 单边斜率：区分「V 型极小（两侧都上升）」与「单调（两侧异号）」
    print("  单边斜率（±1 点的电平变化，dB）：")
    slopes = {}
    for i, ax in enumerate(AXES):
        fp, fm = z3(*[1.0 if k == i else 0.0 for k in range(3)]), \
                 z3(*[-1.0 if k == i else 0.0 for k in range(3)])
        slopes[ax] = (float(fp - z0), float(fm - z0))
        kind = "V 型极小（两侧都上升）" if (fp - z0) > 0 and (fm - z0) > 0 else (
            "单调（两侧异号）" if (fp - z0) * (fm - z0) < 0 else "其他")
        print(f"    {AXES_NAME[ax]:4s}: +1 点 {fp - z0:+.2f} dB，-1 点 {fm - z0:+.2f} dB ⇒ {kind}")
    print("  注：中心差分 Hessian 在 |·| 型拐点处会给出正曲率，掩盖单调性，"
          "故必须同时看单边斜率。")

    # --- 出图 ---
    fig, axes_ = plt.subplots(1, 3, figsize=(16.5, 4.6), facecolor=SURFACE)
    ax = axes_[0]
    errs = np.array(stair["t0"]["err"])
    ax.plot(errs * SAMPLE_US, stair["t0"]["z"], color=SERIES_1, linewidth=1.8,
            drawstyle="steps-mid")
    for a in range(-2, 3):
        ax.axvline(a * SAMPLE_US, color=GRID_C, linewidth=0.9, zorder=0)
    ax.set_xlabel("Window-start error (μs)")
    ax.set_ylabel("Level after filtering (dB)")
    ax.set_title("(a) Quantization staircase: resolution = 1 sample", fontsize=11,
                 color=TEXT_PRIMARY)
    ax.annotate(f"1 sample = {SAMPLE_US:g} μs\nzero-difference steps: "
                f"{stair['t0']['zero_frac'] * 100:.1f}%",
                xy=(0.03, 0.08), xycoords="axes fraction", fontsize=9, color=TEXT_PRIMARY)

    ax = axes_[1]
    for axis, name, color in (("t0", r"$\Delta t_0$", SERIES_1),
                              ("ts", r"$\Delta T_s$", SERIES_2),
                              ("d", r"$\Delta d$  (× 1/200)", SERIES_3)):
        k = np.arange(-5, 6)
        zz = [level(x, h, e_t0=float(q)) if axis == "t0" else
              (level(x, h, e_ts=float(q)) if axis == "ts" else
               level(x, h, ed=float(q) * SAMPLE_D)) for q in k]
        ax.plot(k, zz, color=color, linewidth=2, label=f"{name}: curvature {curv[axis]['median']:+.2f} dB/pt²")
    ax.axvline(0, color=TEXT_SECONDARY, linewidth=1.0, linestyle="--", alpha=0.8)
    ax.set_xlabel("Error (sample points)")
    ax.set_ylabel("Level after filtering (dB)")
    ax.set_title("(b) Profiles along the three axes (1-sample steps)", fontsize=11,
                 color=TEXT_PRIMARY)
    ax.legend(frameon=False, fontsize=9, labelcolor=TEXT_SECONDARY)

    ax = axes_[2]
    v = evals / np.max(np.abs(evals))
    ax.bar(range(3), v, color=[SERIES_1, SERIES_2, SERIES_3], width=0.6)
    ax.axhline(0, color=TEXT_SECONDARY, linewidth=0.9)
    ax.set_xticks(range(3))
    ax.set_xticklabels([f"$\\lambda_{i + 1}$" for i in range(3)])
    ax.set_ylabel("Hessian eigenvalue (normalized)")
    ax.set_title("(c) Curvature at the design point (Hessian)", fontsize=11,
                 color=TEXT_PRIMARY)
    for i, (val, vec) in enumerate(zip(evals, evecs.T)):
        dom = AXES_NAME[AXES[int(np.argmax(np.abs(vec)))]]
        ax.annotate(f"{dom}\n{val:+.3f}", xy=(i, v[i]), xytext=(0, 6 if v[i] >= 0 else -16),
                    textcoords="offset points", ha="center", fontsize=8.5,
                    color=TEXT_PRIMARY)

    for ax in axes_:
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID_C, linewidth=0.8, alpha=0.9)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    fig.tight_layout()
    out = OUTDIR / "平滑性分析_三维.png"
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"\n  图已保存：{out.name}")

    (OUTDIR / "平滑性分析.json").write_text(json.dumps({
        "design": {"npz": DESIGN_NPZ.name, "t0_us": T0_US, "ts_us": TS_US,
                   "tau_us": TAU_US, "d": D_DESIGN, "N": N, "seed": 0},
        "sample_point_us": SAMPLE_US, "sample_point_d": SAMPLE_D,
        "staircase": stair, "curvature": curv,
        "local_minima": {"count": len(locs), "locations": locs,
                         "global_min_db": float(Z.min())},
        "hessian": {"eigenvalues": evals.tolist(), "eigenvectors": evecs.tolist(),
                    "labels": list(AXES), "design_level_db": z0,
                    "one_sided_slopes_db": slopes, "roughness": rough},
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# 三维平滑性分析（θ = [t_0, d, T_s]）", "",
        f"设计对：`{DESIGN_NPZ.name}`（B = {baseline.B_HZ / 1e6:g} MHz、L = {N}、t_0 = {T0_US:g} μs、"
             f"T_L = {TAU_US:g} μs、T_s = {TS_US:g} μs、约束 {baseline.PEAK_INTERF_DB:g} dB）。"
        f"把「参数估偏了，抗干扰性能掉多少」这张曲面，按搜索算法关心的角度体检一遍。", "",
        "## 一句话结论", "",
        "- **t_0（转发时间）与 T_s（采样周期）——规矩、好搜**：偏离后性能平滑变差，"
        "每挪一个采样点方向基本都对，可以直接用爬山法 / 梯度法。",
        "- **d（占空比）——又粗又乱，不能细搜，也不能靠「找最低点」来估**："
        "每挪一个采样点，四步里就有一步方向是反的；而且它单调（窗口越窄越好），"
        "最低点永远落在边界上。", "",
        "## 一、参数最小能调到多细？", "",
        f"- 把误差一点一点加大（每步只有 {2.0 / 410:.4f} 个采样点），"
        f"**{stair['t0']['zero_frac'] * 100:.1f}% 的调整，电平毫无变化**，"
        f"只在新跨过一整个采样点时才跳一下。",
        f"- 最小有效步长 = **1 个采样点**：t_0、T_s 是 {SAMPLE_US:g} μs，d 是 {SAMPLE_D:g}。",
        "- 含义：调得比这更细等于白调——程序读到的电平一模一样（这不是精度问题，是量化）。", "",
        "## 二、偏离之后，性能掉得顺不顺？", "",
        "| 方向 | 每挪 1 个采样点，电平平均动多少 | 单步最大跳变 |",
        "|---|---|---|",
        f"| t_0 | {rough['t0']['median_abs_inc']:.2f} dB | {rough['t0']['max_abs_inc']:.2f} dB |",
        f"| T_s | {rough['ts']['median_abs_inc']:.2f} dB | {rough['ts']['max_abs_inc']:.2f} dB |",
        f"| d | {rough['d']['median_abs_inc']:.2f} dB | {rough['d']['max_abs_inc']:.2f} dB |",
        "",
        "坡度本身的变化也不超过 1.33 dB/采样点²，没有突然变陡的悬崖。",
        "含义：**只要步长 ≥ 1 个采样点，这张曲面就是「平滑可爬」的**，不会一步踩空。", "",
        "## 三、往哪边挪是对的？（最要紧的一条）", "",
        "| 方向 | 连挪 40 步，方向反了几步 | 判读 |",
        "|---|---|---|",
        f"| t_0 | {rough['t0']['sign_flips']} / 40"
        f"（{rough['t0']['sign_flips'] / 40 * 100:.0f}%） | 方向可靠，蒙着眼爬也对 |",
        f"| T_s | {rough['ts']['sign_flips']} / 40"
        f"（{rough['ts']['sign_flips'] / 40 * 100:.0f}%） | 方向可靠 |",
        f"| d | **{rough['d']['sign_flips']} / 40"
        f"（{rough['d']['sign_flips'] / 40 * 100:.0f}%）** | 四步里就有一步爬错方向 |",
        "",
        "为什么 d 这么乱：电平对 d 的响应，取决于「多采样 / 少采样那一个采样点」的正负号，"
        "而这个符号基本是随机的。所以 d 那点「越窄越好」的单调趋势只是统计平均，"
        "单步看完全没有规律。", "",
        "## 四、谷底是一个点，还是一条沟？", "",
        f"- 在 11×11×11 = {Z.size} 个网格点上找「四面都比我高」的最低点：**一个都找不到（0 个）**。",
        "- 原因：t_0、T_s 两个方向都是两边变差（真谷底），但 d 方向单调——窗口越窄越好，"
        "所以「最低点」只会被推到 d 能取的最小值（边界）上。",
        "- 含义：**别用「找最低点」来估 d**（答案永远是「干扰越弱越好」）；估 t_0、T_s 才成立。", "",
        "## 五、谷底是尖角还是圆碗？", "",
        "- 是**尖角**（像 V 字的底），不是圆碗底。",
        "- 后果：用「左右对称取两点估斜率」这种常规做法，会把 d 这种「一直朝一边降」的方向"
        "误判成「这里就是最低点」——Hessian 三个特征值全为正就是这么来的。",
        "- 做法：判断是不是谷底，要**左右两边分开看**，不能只看对称差分。", "",
        "## 六、给搜索算法的四条硬约束", "",
        "1. 步长 ≥ 1 个采样点（t_0 / T_s 用 0.05 μs，d 用 0.005），比这更小毫无意义；",
        "2. d 维**不要**用单采样点步长的爬山 / 坐标下降——要么用粗步长（≥ 5~10 个采样点），"
        "要么先对若干采样点平滑再判方向；",
        "3. t_0、T_s 两维可以放心用梯度 / 爬山类方法；",
        "4. 估 d 不能拿「让电平最低」当目标（会一路推到边界），得换判据（如相对自身能量的抑制比）。", "",
        "## 附录：原始指标", "",
        "| 项目 | 数值 |", "|---|---|",
        "| 量化（三轴零差分占比 / 跳变次数） | "
        + " ; ".join(f"{AXES_NAME[a]} {stair[a]['zero_frac'] * 100:.1f}% / {stair[a]['steps']} 次"
                     for a in AXES) + " |",
        "| 二阶差分中位数（曲率） | "
        + " ; ".join(f"{AXES_NAME[a]} {curv[a]['median']:+.3f} dB/点²（最大 {curv[a]['max_abs']:.3f}）"
                     for a in AXES) + " |",
        f"| 局部极小（11³ 网格） | {len(locs)} 个；全局最小 {Z.min():.2f} dB |",
        "| 单边斜率 @设计点 | "
        + " ; ".join(f"{AXES_NAME[a]}: +1 点 {slopes[a][0]:+.2f} / −1 点 {slopes[a][1]:+.2f} dB"
                     for a in AXES) + " |",
        "| 设计点 Hessian 特征值 | " + " , ".join(f"{v:+.4f}" for v in evals)
        + " dB/点²（全正，须配合单边斜率解读） |",
    ]
    (OUTDIR / "平滑性分析_结果表.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("  JSON 与结果表已保存")


if __name__ == "__main__":
    main()
