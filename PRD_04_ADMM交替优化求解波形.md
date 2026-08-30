# PRD 模块四：ADMM 交替优化求解波形

> 对应论文 Section III，式 (9)–(16)
> **核心模型之二**：给定失配滤波器 h，使用 ADMM 求解非凸波形优化子问题

---

## 1. 模块目标

在交替优化框架中，当失配滤波器 `h` 固定时，波形设计子问题因 PAPR 约束而**非凸**，无法直接求解析解。本模块引入辅助变量 `y, z`，将非凸约束解耦，使用**交替方向乘子法（ADMM）**迭代求解，保证在合适步长下收敛。

---

## 2. 波形子问题表述（式 11）

### 2.1 原始问题

给定固定滤波器 `h`，求解：

```
min_{x ∈ C^N}   x^H P x
s.t.
    h^H x = b_max          （主瓣峰值约束）
    h_j^H x = b_min        （峰值干扰约束）
    max_n |x[n]|^2 / (||x||^2 / N) ≤ η   （PAPR 约束）
```

其中 `h_j = h ⊙ j`，`P = ε H^H D H + (1-ε) H_j^H H_j`（由模块二构造）。

### 2.2 PAPR 约束的等价改写（式 9–10）

论文将 PAPR 约束改写为两个等价约束：

```
x^H x = N                  （总能量约束）
x^H E_n x ≤ η,  n = 1,...,N   （逐元素模长约束）
```

其中 `E_n` 是 N×N 矩阵，仅第 n 个对角元为 1，其余为 0：

```
E_n(i, j) = 1   if i = j = n
           = 0   otherwise
```

`x^H E_n x = |x[n]|^2`，因此第二组约束等价于 `|x[n]|^2 ≤ η, ∀n`。

当 `||x||^2 = N` 时，`max_n |x[n]|^2 / (||x||^2/N) = max_n |x[n]|^2 ≤ η`，与标准 PAPR 定义一致。

### 2.3 非凸性来源

- 目标函数 `x^H P x` 是凸的（P 半正定）
- 线性等式约束是凸的
- **`|x[n]|^2 ≤ η` 是凸的**（每个元素模长限制）
- 但 `x^H x = N`（球面约束）是**非凸**的

因此整体问题非凸，ADMM 通过引入辅助变量将非凸约束隔离到简单子问题中。

---

## 3. 辅助变量与增广拉格朗日函数

### 3.1 辅助变量引入

引入两个辅助变量 `y, z ∈ C^N`，将约束重写为：

```
x = y,   s.t. y^H y = N          （总能量约束交给 y）
x = z,   s.t. |z[n]|^2 ≤ η, ∀n  （PAPR 约束交给 z）
```

### 3.2 增广拉格朗日函数（scaled form，式 12）

```
L(x, y, z, λ, γ, μ, ν) = x^H P x
    + (ρ/2) (||x - z + λ||^2 - ||λ||^2)
    + (ρ/2) (||x - y + γ||^2 - ||γ||^2)
    + (ρ/2) (|h^H x - b_max + μ|^2 - |μ|^2)
    + (ρ/2) (|h_j^H x - b_min + ν|^2 - |ν|^2)
```

其中：
- `ρ > 0`：惩罚参数 / 步长（默认 5000）
- `λ, γ ∈ C^N`：对应 `x=z`、`x=y` 的 scaled 对偶变量
- `μ, ν ∈ C`（标量）：对应 `h^H x=b_max`、`h_j^H x=b_min` 的 scaled 对偶变量

---

## 4. ADMM 迭代步骤（式 13a–13d）

### 4.1 步骤 1：更新 x（式 13a，闭式解式 14）

固定 `y_t, z_t, λ_t, γ_t, μ_t, ν_t`，最小化 `L` 关于 `x`：

```
x_{t+1} = arg min_x L(x, y_t, z_t, λ_t, γ_t, μ_t, ν_t)
```

**推导**：将复数 `x` 拆为实部+虚部，对实向量求导并令为零（系数因此为 2P），整理得线性方程组：

```
A · x_{t+1} = b_vec
```

其中系数矩阵：
```
A = 2P + 2ρI + ρ h h^H + ρ h_j h_j^H    （N×N Hermitian 矩阵）
```

右侧向量：
```
b_vec = ρ(z_t - λ_t) + ρ(y_t - γ_t)
      + ρ(b_max - μ_t) h + ρ(b_min - ν_t) h_j
```

**闭式解**：
```
x_{t+1} = A^{-1} b_vec
```

**实现要点**：
- `A` 是 Hermitian 正定矩阵（2P 半正定 + 2ρI 正定 + 半正定项），可用 Cholesky 分解高效求解
- `h h^H` 和 `h_j h_j^H` 是秩-1 外积，构造时注意不要用循环
- `P` 依赖于当前 `h`（H 和 Hj 由 h 构造），每次外层迭代更新 h 后需重新构造 P

### 4.2 步骤 2：更新 y（式 13b，闭式解式 15）

固定 `x_{t+1}, γ_t`，最小化关于 `y`：

```
y_{t+1} = arg min_y ||x_{t+1} - y + γ_t||^2
s.t. y^H y = N
```

**解析解**：将 `(x_{t+1} + γ_t)` 投影到半径为 `√N` 的复球面上：

```
y_{t+1} = √N × (x_{t+1} + γ_t) / ||x_{t+1} + γ_t||
```

即保持方向不变，将范数归一化为 `√N`。

**论文笔误说明**：论文式 (13b) 中目标函数写为 `||x^{t+1} - y + λ^t||²`（用了 λ），但式 (15) 的解析解中用的是 `γ^t`。根据增广拉格朗日函数 (12)，`x-y` 约束对应对偶变量 `γ`，`x-z` 约束对应 `λ`。因此**式 (13b) 的 λ 是论文笔误，应为 γ**。本 PRD 采用正确的 γ。

**特殊情况**：若 `||x_{t+1} + γ_t|| ≈ 0`（数值下溢），可设 `y` 为随机单位模向量乘以 `√N`。

### 4.3 步骤 3：更新 z（式 13c，闭式解式 16）

固定 `x_{t+1}, λ_t`，最小化关于 `z`：

```
z_{t+1} = arg min_z ||x_{t+1} - z + λ_t||^2
s.t. |z[n]|^2 ≤ η,  n = 1,...,N
```

**解析解**：逐元素裁剪（element-wise clipping）。令 `w = x_{t+1} + λ_t`，对每个元素 n：

```
z_{t+1}[n] = w[n]                          if |w[n]|^2 ≤ η
            = √η × w[n] / |w[n]|            if |w[n]|^2 > η
```

即：若元素模长不超过 `√η`，保持不变；若超过，则将模长压缩到 `√η`，**相位保持不变**。

这是利用 KKT 条件得到的逐元素闭式解，无需迭代。

### 4.4 步骤 4：更新对偶变量（式 13d）

```
λ_{t+1} = x_{t+1} - z_{t+1} + λ_t
γ_{t+1} = x_{t+1} - y_{t+1} + γ_t
μ_{t+1} = h^H x_{t+1} - b_max + μ_t
ν_{t+1} = h_j^H x_{t+1} - b_min + ν_t
```

均为 scaled form 的标准对偶变量更新。

---

## 5. 收敛判据与停止条件

### 5.1 内层 ADMM 收敛

ADMM 内层迭代停止条件（论文原图确认）：

```
||x_{t+1} - y_{t+1}||_2 + ||x_{t+1} - z_{t+1}||_2 < δ
```

其中 `δ = 1e-4`（论文设定）。

**注意**：这是两个 2-范数的**和**（不是平方和）。论文报告最终迭代的 consensus error = 8.24e-5，满足 < 1e-4。

### 5.2 内层最大迭代次数

论文未明确给出 ADMM 内层最大迭代次数。为防止死循环，设置：
```
max_inner_iter = 500（默认，可配置）
```

达到最大迭代次数即使未收敛也停止。

### 5.3 收敛性说明

论文引用 [18], [20], [21] 证明：对于非凸问题，ADMM 在**合适步长 ρ** 下保证收敛。本文取 `ρ = 5000`。步长过小可能不收敛，过大则收敛慢。

---

## 6. 完整 ADMM 伪代码

```python
def admm_waveform_optimization(h, j, P, b_max, b_min, eta, rho, delta,
                                 max_inner_iter, x_init, y_init, z_init,
                                 lambda_init, gamma_init, mu_init, nu_init):
    """
    ADMM求解波形优化子问题（给定h）
    """
    h_j = h * j                          # 滤波器与采样函数逐元素乘
    x, y, z = x_init, y_init, z_init
    lam, gam, mu, nu = lambda_init, gamma_init, mu_init, nu_init

    # 预计算系数矩阵 A（在ADMM内层迭代中A不变，因为h固定）
    A = 2*P + 2*rho*np.eye(N) + rho*np.outer(h, h.conj()) + rho*np.outer(h_j, h_j.conj())
    # Cholesky分解 A = R^H R，用于多次求解
    R = np.linalg.cholesky(A)

    for t in range(max_inner_iter):
        # 步骤1: 更新x
        b_vec = rho*(z - lam) + rho*(y - gam) + rho*(b_max - mu)*h + rho*(b_min - nu)*h_j
        x = solve_cholesky(R, b_vec)    # x = A^{-1} b_vec

        # 步骤2: 更新y
        w_y = x + gam
        y = np.sqrt(N) * w_y / np.linalg.norm(w_y)

        # 步骤3: 更新z（逐元素裁剪，加epsilon防止除零）
        w_z = x + lam
        abs_wz = np.abs(w_z)
        z = np.where(abs_wz**2 <= eta, w_z, np.sqrt(eta) * w_z / (abs_wz + 1e-12))

        # 步骤4: 更新对偶变量
        lam = x - z + lam
        gam = x - y + gam
        mu = np.vdot(h, x) - b_max + mu
        nu = np.vdot(h_j, x) - b_min + nu

        # 收敛判据（两个2范数的和，不是平方和）
        consensus_error = np.linalg.norm(x - y) + np.linalg.norm(x - z)
        if consensus_error < delta:
            break

    return x, y, z, lam, gam, mu, nu
```

---

## 7. 接口规范

```python
def solve_waveform_admm(
    h: np.ndarray,
    j: np.ndarray,
    P: np.ndarray,
    params: GlobalParams,
    variables: dict
) -> tuple:
    """
    ADMM求解波形子问题

    输入:
        h: 失配滤波器, shape=(N,)
        j: ISRJ采样函数, shape=(N,)
        P: 加权矩阵, shape=(N,N)
        params: 全局参数（含rho, delta, eta, b_max, b_min, max_inner_iter）
        variables: 包含x, y, z, lambda, gamma, mu, nu的初始值

    输出:
        x_new: 优化后的波形, shape=(N,)
        variables_updated: 更新后的所有变量
        info: 迭代次数、最终consensus error等调试信息
    """
```

---

## 8. 验收标准

1. **x 更新正确性**：每次 x 更新满足 `A x ≈ b_vec`（残差 < 1e-8）。
2. **y 约束满足**：`|y^H y - N| / N < 1e-6`。
3. **z 约束满足**：`max_n |z[n]|^2 ≤ η + 1e-6`（所有元素模方不超过 η）。
4. **对偶变量更新**：严格按式 (13d) 更新，无遗漏。
5. **收敛性**：对于随机初始化，在 `max_inner_iter` 内 consensus error（两个2范数的和）降至 `δ=1e-4` 以下（论文报告最终迭代 consensus error = 8.24e-5）。
6. **目标函数下降**：ADMM 迭代过程中增广拉格朗日函数值（或原目标函数 `x^H P x`）总体呈下降趋势。
7. **PAPR 约束**：最终输出的 x 满足 `max|x[n]|^2 / (||x||^2/N) ≤ η`（允许微小数值误差）。
8. **步长敏感性**：`ρ=5000` 时稳定收敛；可测试不同 ρ 观察收敛行为。

---

## 9. 与论文的对应关系

| 论文元素 | 本模块实现 |
|---------|-----------|
| 式 (9) PAPR 约束改写 | `x^H x=N` + `|x[n]|^2≤η` |
| 式 (10) E_n 矩阵定义 | 单位矩阵的第 n 列 |
| 式 (11) 波形子问题 | 带辅助变量的非凸优化 |
| 式 (12) 增广拉格朗日函数 | scaled form，含 4 组约束 |
| 式 (13a) x 更新 | 闭式解，Cholesky 求解 |
| 式 (13b) y 更新 | 球面投影 |
| 式 (13c) z 更新 | 逐元素裁剪 |
| 式 (13d) 对偶变量更新 | 4 个变量同步更新 |
| 式 (14) x 闭式解 | `A^{-1} b_vec`，A=2P+2ρI+ρhh^H+ρh_jh_j^H |
| 式 (15) y 闭式解 | `√N × (x+γ)/||x+γ||` |
| 式 (16) z 闭式解 | 逐元素模长裁剪到 √η |
| "alternating direction multiplier method (ADMM)" | 模块核心方法 |
| "non-convex subproblem" | PAPR + 球面约束导致非凸 |
| "suitable step size ensuring the convergence" | ρ=5000 |

---

## 10. 已确认事项

1. **ADMM 内层最大迭代次数 = 500**：论文未明确给出，取 500。**已确认。**
2. **系数矩阵 A 中的 2P**：论文式 (14) 系数为 `2P + 2ρI + ...`。这是将复数变量拆为实部+虚部后对实向量求导的结果。**实现时必须用 2P，不能用 P。已确认。**
3. **收敛判据为范数和**：论文原图确认 consensus error = `||x-y||₂ + ||x-z||₂ < δ`（两个2范数的和，不是平方和）。**已确认。**
4. **μ 和 ν 的复数性**：约束 `h^H x = b_max`（b_max 为实数）是复数等式约束，μ 应为复数。实现时按复数处理。**已确认。**
5. **P 的更新时机**：P 依赖于 h（H 和 Hj 由 h 构造）。在 ADMM 内层迭代中 h 固定，P 不变；在外层交替优化中 h 更新后需重新构造 P。**已确认。**
