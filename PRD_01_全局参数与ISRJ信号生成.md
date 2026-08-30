# PRD 模块一：全局参数初始化与 ISRJ 信号生成

> 论文：*Joint Design of Transmit Waveform and Mismatch Filter in the Presence of Interrupted Sampling Repeater Jamming*
> 作者：Kai Zhou, Dexin Li, Yi Su, Tao Liu (IEEE Signal Processing Letters, 2020)
> DOI: 10.1109/LSP.2020.3021667

---

## 1. 模块目标

本模块负责：
1. 集中定义并初始化所有全局仿真参数（雷达参数、干扰机参数、算法参数、约束参数）。
2. 生成 ISRJ（中断采样转发干扰）的离散采样函数 `j`。
3. 生成初始波形 `x`、初始失配滤波器 `h`、辅助变量和对偶变量。

所有下游模块均从本模块读取参数，**禁止在其他模块中硬编码参数**。

---

## 2. 全局参数表

### 2.1 雷达信号参数

| 参数符号 | 含义 | 取值 | 单位 | 备注 |
|---------|------|------|------|------|
| `T_p` | 信号脉冲持续时间 | 40 | μs | 论文 Section IV |
| `B` | 信号带宽 | 20 | MHz | 论文 Section IV |
| `f_c` | 雷达载频 | 1 | GHz | 论文 Table I |
| `N` | 波形码长 / 采样点数 | **800** | — | `N = B × T_p = 20e6 × 40e-6 = 800`（复采样，采样率=带宽）。**已确认**：论文图 1(b) 横轴到 800，验证 N=800 |
| `fs` | 采样率 | `B`（=20） | MHz | 复采样默认等于带宽 |

### 2.2 干扰机参数

| 参数符号 | 含义 | 取值 | 单位 | 备注 |
|---------|------|------|------|------|
| `T_s` | 干扰机采样重复周期 | 10 | μs | 论文 Section IV / Table I |
| `tau` | 中断采样脉冲宽度 | 2.5 | μs | 论文 Section IV / Table I |
| `tau_delay_jammer` | 干扰机转发时延 | 2 | μs | 论文 Table I（第三个仿真用） |
| `num_repeats` | 干扰机重发次数 | 2 | — | 论文 Section IV："jammer retransmits the interrupted sampling signal twice"（第三个仿真） |
| `SJR` | 信干比 | -10 | dB | 论文 Section IV（第三个仿真） |
| `pos_radar` | 雷达位置 | (0, 0, 0) | m | 论文 Table I |
| `pos_jammer` | 干扰机位置 | (0, 2000, 0) | m | 论文 Table I |
| `pos_target` | 目标位置 | (0, 3000, 0) | m | 论文 Table I |

### 2.3 算法超参数

| 参数符号 | 含义 | 取值 | 单位 | 备注 |
|---------|------|------|------|------|
| `M` | 外层最大迭代次数（交替优化轮数） | 100 | — | 论文 Section IV |
| `delta` | ADMM 内层收敛阈值（consensus error） | 1e-4 | — | 论文 Section IV：`||x-y||₂ + ||x-z||₂ < δ`（两个2范数的和） |
| `rho` | ADMM 惩罚参数 / 步长 | 5000 | — | 论文 Section IV。需足够大以保证非凸问题收敛 |
| `epsilon` | Pareto 权重 | 0.5 | — | 论文 Section IV。平衡 ISLs 与 ILs |
| `max_inner_iter` | ADMM 内层最大迭代次数 | 500 | — | 论文未明确给出，需设置以防止死循环。**已确认**：取 500 |

### 2.4 约束参数

| 参数符号 | 含义 | 取值 | 单位 | 备注 |
|---------|------|------|------|------|
| `b_max` | 主瓣峰值约束 | `N` | — | 论文 Section IV：`bmax = N` |
| `peak_interf_db` | 归一化峰值干扰限制 | -30 | dB | 论文 Section IV |
| `b_min` | 峰值干扰约束值 | `N × 10^(-1.5)` | — | 由 `20lg(bmin/bmax) = -30dB` 推导：`bmin = bmax × 10^(-30/20) = N × 10^(-1.5)` |
| `eta` | PAPR 约束值 | 1.1（默认） | — | 论文第一个仿真 η=1.1；第二个仿真取 {1, 1.05, 1.1, 1.15, 1.21, 1.44} |
| `PAPR_range` | PAPR 扫描范围 | [1, 1.05, 1.1, 1.15, 1.21, 1.44] | — | 论文第二个仿真 |

---

## 3. ISRJ 采样函数 `j` 的生成

### 3.1 物理原理

ISRJ 干扰机对雷达发射信号进行间歇采样后转发。采样函数为矩形脉冲序列：
- 周期：`T_s = 10 μs`
- 脉宽：`tau = 2.5 μs`
- 占空比：`tau / T_s = 0.25`

### 3.2 离散化

将连续时间矩形脉冲序列按采样率 `fs` 离散化为长度为 `N` 的向量 `j ∈ {0, 1}^N`。

**采样点数换算**：
- 周期内采样点数：`N_Ts = round(T_s × fs) = round(10e-6 × 20e6) = 200`
- 脉宽内采样点数：`N_tau = round(tau × fs) = round(2.5e-6 × 20e6) = 50`

**生成规则**：
```
for n = 0, 1, ..., N-1:
    if mod(n, N_Ts) < N_tau:
        j[n] = 1
    else:
        j[n] = 0
```

即每个周期的前 `N_tau` 个采样点为 1（采样），其余为 0（不采样）。

### 3.3 输出

- `j`: 长度为 `N` 的实向量，元素为 0 或 1。
- `j` 用于构造 `h_j = h ⊙ j`（失配滤波器与采样函数的 Hadamard 积），以及 `x_j = x ⊙ j`（ISRJ 信号）。

---

## 4. 变量初始化

### 4.1 波形 `x`

- 类型：复向量 `x ∈ C^(N×1)`
- 初始化：随机相位序列（random-phase sequence）
- 生成方式：`x[n] = exp(j × 2π × rand())`，即单位模随机相位
- 归一化：确保 `||x||^2 = N`（初始即满足，因为每个元素模长为 1）

### 4.2 失配滤波器 `h`

- 类型：复向量 `h ∈ C^(N×1)`
- 初始化：随机相位序列
- 生成方式：同 `x`，`h[n] = exp(j × 2π × rand())`

### 4.3 辅助变量

| 变量 | 维度 | 初始化 | 约束 |
|------|------|--------|------|
| `y` | C^(N×1) | 复制 `x` 的初始值 | `y^H y = N` |
| `z` | C^(N×1) | 复制 `x` 的初始值 | `|z[n]|^2 ≤ η, ∀n` |

### 4.4 对偶变量（scaled form）

论文原文：*"The waveform, mismatch filter, auxiliary and dual variables are initialized with random-phase sequences."* 即**所有变量（含对偶变量）均用随机相位序列初始化**。

| 变量 | 维度 | 初始化 | 对应约束 |
|------|------|--------|---------|
| `λ` (lambda) | C^(N×1) | 随机相位单位模向量 | `x = z` |
| `γ` (gamma) | C^(N×1) | 随机相位单位模向量 | `x = y` |
| `μ` (mu) | 标量（复数） | 随机相位单位模复数 `exp(jθ)` | `h^H x = b_max` |
| `ν` (nu) | 标量（复数） | 随机相位单位模复数 `exp(jθ)` | `h_j^H x = b_min` |

**说明**：ADMM 中对偶变量通常初始化为零，但论文明确指定用随机相位序列初始化。scaled form 下对偶变量的初始值不影响算法收敛性，仅影响收敛路径。实现时按论文要求用随机相位初始化。

---

## 5. 接口规范

### 5.1 输出数据结构

```python
class GlobalParams:
    # 雷达参数
    T_p: float = 40e-6
    B: float = 20e6
    f_c: float = 1e9
    N: int = 800
    fs: float = 20e6

    # 干扰机参数
    T_s: float = 10e-6
    tau: float = 2.5e-6
    tau_delay_jammer: float = 2e-6
    num_repeats: int = 2
    SJR_db: float = -10

    # 算法参数
    M: int = 100
    delta: float = 1e-4
    rho: float = 5000
    epsilon: float = 0.5
    max_inner_iter: int = 500

    # 约束参数
    b_max: float = None  # = N
    b_min: float = None  # = N * 10^(-1.5)
    eta: float = 1.1

    def __post_init__(self):
        self.b_max = self.N
        self.b_min = self.N * 10 ** (-1.5)
```

### 5.2 函数接口

```python
def generate_isrj_sampling_function(params: GlobalParams) -> np.ndarray:
    """
    生成 ISRJ 离散采样函数 j
    返回: j, shape=(N,), dtype=float, 元素为0或1
    """

def initialize_variables(params: GlobalParams) -> dict:
    """
    初始化 x, h, y, z, lambda, gamma, mu, nu
    返回: 包含所有变量的字典
    """
```

---

## 6. 验收标准

1. `j` 的长度为 `N`，元素仅为 0 或 1，每个周期内前 `N_tau` 个为 1。
2. `j` 的占空比 ≈ `tau / T_s = 0.25`。
3. 初始 `x` 和 `h` 为单位模随机相位向量，`||x||^2 = N`，`||h||^2 = N`。
4. 初始 `y` 满足 `y^H y = N`，初始 `z` 满足 `|z[n]|^2 ≤ η, ∀n`。
5. 所有对偶变量初始化为随机相位单位模（向量或标量），与论文要求一致。
6. 参数集中管理，无硬编码。

---

## 7. 已确认参数与说明

1. **码长 N = 800**：按复采样（fs=B=20MHz）得 N=20e6 × 40e-6 = 800。论文图 1(b) 横轴到 800 验证此值。**已确认。**
2. **ADMM 内层最大迭代次数 = 500**：论文未明确给出，取 500 防止死循环。**已确认。**
3. **干扰机时延与重发的建模**：联合设计阶段（图 1、图 2、图 3）使用单周期无延迟采样函数 `j`；仅在图 4 脉冲压缩验证中引入 2μs 时延和两次重发。图 4 暂不实现。**已确认。**
