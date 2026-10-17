# 结果目录：两条线的分工

**2026-10-04 重组**。本目录下只有两个文件夹，各管一件事：

| 文件夹 | 管什么 | 什么时候来这儿 |
|---|---|---|
| **`参数调制/`** | **调干扰机 / 雷达信号的参数**——误差扫描、代价函数 C、V 型谷、联合误差图、参数空间探索 | "我想看某个参数变了会怎样" |
| **`对抗场景/`** | **把参数在雷达对抗过程中体现**——幅度-距离图、联合波形/失配滤波设计、参数失配、BO 参数搜索 | "我想看这在雷达屏幕上是什么样" |

---

## 参数唯一来源（保证两边一致）

两条线的参数**都从 `../scripts/baseline.py` 取**，没有任何一处硬编码：

```
baseline.py（v2 基线，对齐 Zhou 2026）
    B = 10 MHz、T_p = 25.6 μs、码长 L(N) = 256、fs = B ⇒ 1 采样点 = 0.1 μs
    t_0 = 2 μs（20 点）、T_L = 1.5 μs（15 点）、T_s = 6.4 μs（64 点）、d = 0.234
    干扰峰值约束 −80 dB（b_min = N·10⁻⁴）

    ↓ 引用方式

参数调制/  ← 脚本直接 import baseline（μs 为单位）
对抗场景/  ← 脚本从 baseline 换算成采样点，例如
             run_scene_error_sweep_3param.py: BASE = {t_0: round(T0_US/0.1), ...}
```

**要改参数只改 `baseline.py` 一处**，两条线同时生效。对抗场景脚本里有断言
校验"自建掩膜 == 项目函数生成的掩膜"，改错了会立刻报错。

> ⚠️ 复现线（仓库根的 `isrj_joint_design/`、`PRD_01~05`）是**另一套体系**
> （N=800、B=20 MHz、约束 −30 dB），不在本目录范围内，也不要与这里混用。

---

## 目录内容

```
结果/
├── 参数调制/
│   ├── 设计对/          共享设计对 npz（各脚本读它，缺失会自动跑一次设计）
│   ├── 联合误差/        三对联合误差图（t_0×T_s、t_0×T_L、T_L×T_s）+ 一维参数误差曲线
│   ├── 平滑性/          三维平滑性分析
│   ├── 拟合/            t_0×T_s 电平地图（121 格点）与双线性插值面
│   ├── 占空比扫描/      占空比一维误差扫描
│   ├── 窗口起点扫描/    t_0 一维误差扫描
│   ├── 采样周期扫描/    T_s 一维误差扫描
│   └── _v1_自建800点/   v1 归档（不再更新）
│
└── 对抗场景/
    ├── scene01_normal_range_profile.png          正常探测
    ├── scene02_jamming_range_profile.png         被 ISRJ 干扰（3 个假目标）
    ├── scene03_before_after_design.png           联合设计前后
    ├── scene04_error_*.png                       参数误差（一维扫描 / 三档距离像）
    ├── scene05_error_3param*.png                 三参数误差（柱状图 / 3×3 距离像）
    ├── scene06_error_joint_profiles.png          三参数同时失配
    ├── scene07_bo_search_*.png                   BO 参数搜索的前后对比
    └── *.json / scene_design_pair.npz            数值记录与设计对
```

## 相关脚本

| 脚本（`../scripts/`） | 产出到 |
|---|---|
| `run_joint_error_maps.py`、`plot_joint_error_summary.py` | `参数调制/联合误差/` |
| `run_param_error_1d.py` | `参数调制/联合误差/`（一维曲线） |
| `run_smoothness_analysis.py` | `参数调制/平滑性/` |
| `run_fit_map_t0_ts.py`、`interp_t0_ts.py` | `参数调制/拟合/` |
| `run_t0_sweep.py`、`run_duty_sweep.py`、`run_ts_sweep.py` | `参数调制/{窗口起点,占空比,采样周期}扫描/` |
| `run_scene_*.py`、`scene_joint_design.py` | `对抗场景/` |

场景线的**口径全文**见 `../docs/对抗场景口径.md`。
