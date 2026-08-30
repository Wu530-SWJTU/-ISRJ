# PRD 模块五：交替优化主算法与相关性绘图

> 对应论文 Section III（Algorithm 1）和 Section IV（Numerical Simulations）
> 整合模块一至四，实现完整联合设计流程，并输出所有仿真图像

---

## 1. 模块目标

1. 实现**交替优化主算法**（Algorithm 1）：外层交替更新滤波器和波形，内层用 ADMM 求解波形子问题。
2. 设计完波形和滤波器后，绘制**无 ISRJ 干扰**和**有 ISRJ 干扰**下的相关性图像（论文图 2）。
3. 复现论文中的三组仿真实验，输出对应图表。

---

## 2. 交替优化主算法（Algorithm 1）

### 2.1 算法流程

```
输入: 全局参数 params, ISRJ采样函数 j
输出: 优化后的波形 x, 失配滤波器 h

1. 初始化:
   - x = 随机相位单位模向量
   - h = 随机相位单位模向量
   - y = x (复制), z = x (复制)
   - λ = 0向量, γ = 0向量, μ = 0, ν = 0
   - 记录目标函数历史: isl_history = [], il_history = []

2. 外层循环 m = 1, 2, ..., M (M=100):
   a. 构造当前 x 对应的矩阵 X, Xj, 组装 Q = ε X^H D X + (1-ε) Xj^H Xj
   b. 固定 x, 调用模块三: h = solve_filter_lagrange(x, j, Q, b_max, b_min)
   c. 构造当前 h 对应的矩阵 H, Hj, 组装 P = ε H^H D H + (1-ε) Hj^H Hj
   d. 固定 h, 调用模块四: (x, y, z, λ, γ, μ, ν) = solve_waveform_admm(h, j, P, params, variables)
   e. 计算当前目标函数:
      - r = H @ x (卷积输出, 长度2N-1)
      - r_j = Hj @ x (ISRJ卷积输出)
      - isl = ||r||^2 - |r[N]|^2  (积分旁瓣电平)
      - il = ||r_j||^2              (ISRJ积分电平)
      - 记录: isl_history.append(isl), il_history.append(il)
   f. 打印/记录迭代信息

3. 输出: x, h, 历史记录
```

### 2.2 关键注意事项

1. **变量热启动（warm start）**：对偶变量 `λ, γ, μ, ν` 和辅助变量 `y, z` 在外层迭代之间保持，不重新初始化。这是 ADMM 收敛的关键。
2. **矩阵更新时机**：
   - `Q` 依赖于 `x`，在步骤 2a（更新 h 之前）构造
   - `P` 依赖于 `h`，在步骤 2c（更新 x 之前）构造
3. **h_j 的构造**：`h_j = h ⊙ j`（逐元素乘），在构造 P 和 ADMM 求解时都需要。
4. **x_j 的构造**：`x_j = x ⊙ j`，在构造 Q 和拉格朗日求解时都需要。
5. **收敛判据**：外层循环固定 M=100 次（论文设定），不设外层收敛判据；内层 ADMM 设收敛判据 `δ=1e-4` 和最大迭代次数。

### 2.3 计算复杂度

- 每次外层迭代：滤波器求解 O(N^3)（Cholesky）+ ADMM 内层迭代 × O(N^3)（每次 x 更新需解线性方程组）
- 总复杂度：O(M × N^3 × inner_iter_avg)
- 论文报告：在 3.2GHz i7 CPU + 64GB RAM 上运行（具体时间未报告）

---

## 3. 相关性图像绘制（核心需求）

### 3.1 无 ISRJ 干扰下的相关性（论文图 2(a)）

**物理意义**：发射波形 `x` 与失配滤波器 `h` 的互相关，即无干扰时的脉冲压缩输出。

**计算方法**：
```
r = xcorr(x, h)  # 互相关, 长度 2N-1
# 或等价地: r = H @ x = X @ h
```

**归一化与 dB 转换**：
```
r_peak = |r[N]|  # 主瓣峰值（第N个元素，0-indexed为N-1）
r_db = 20 * log10(|r| / r_peak)
```

**绘图规格**：
- 横轴：`k = 0, 1, ..., 2N-2`（延迟采样点），或转换为距离 `range = k * c / (2 * fs)`
- 纵轴：归一化互相关电平（dB），范围建议 [-60, 0] dB
- 标题："Cross-correlation between transmitted waveform and mismatched filter"
- 标注主瓣峰值位置和峰值旁瓣电平（PSL）
- 标注处理增益损失：`loss = 10 log10(|h^H x|^2 / (||h||^2 ||x||^2))`，论文报告 -0.9dB

### 3.2 有 ISRJ 干扰下的相关性（论文图 2(b)）

**物理意义**：ISRJ 信号 `x_j = x ⊙ j` 与失配滤波器 `h` 的互相关，即干扰信号经过滤波器后的输出。

**计算方法**：
```
x_j = x * j  # 逐元素乘
r_j = xcorr(x_j, h)  # 互相关, 长度 2N-1
# 或等价地: r_j = Hj @ x = Xj @ h
```

**归一化与 dB 转换**：
```
# 使用与图2(a)相同的归一化因子（波形主瓣峰值），以公平对比
r_j_db = 20 * log10(|r_j| / r_peak)
```

**绘图规格**：
- 横轴：同图 2(a)
- 纵轴：归一化互相关电平（dB），范围建议 [-400, 0] dB（ISRJ 被抑制到很低）
- 标题："Cross-correlation between ISRJ signal and mismatched filter"
- 标注零延迟处的峰值（应 ≤ -30dB，即 `b_min / b_max` 对应的电平）
- 论文图 2(b) 中 ISRJ 峰值约 -30dB，其余旁瓣低至 -300dB 以下

### 3.3 双图对比布局

```
+------------------------+------------------------+
|                        |                        |
|   图(a): 无ISRJ相关性   |   图(b): 有ISRJ相关性   |
|   (波形 vs 滤波器)      |   (ISRJ vs 滤波器)     |
|                        |                        |
+------------------------+------------------------+
```

- 两个子图并排（1×2 布局）
- 共用横轴范围
- 纵轴范围可不同（图a范围小，图b范围大）
- 均使用 dB 刻度

---

## 4. 其他仿真图表（论文复现）

### 4.1 图 1(a)：目标函数收敛曲线

**内容**：随外层迭代次数变化的 ISLs（波形滤波器输出积分旁瓣电平）和 ILs（ISRJ 信号滤波器输出积分电平）。

**绘图**：
- 横轴：迭代次数（0 到 M）
- 纵轴：ISLs / ILs（线性或 dB）
- 两条曲线：ISLs of waveform filter output、ILs of ISRJ signal filter output
- 标注最终收敛值

### 4.2 图 1(b)：设计波形的实部

**内容**：优化后波形 `x` 的实部随采样点变化。

**绘图**：
- 横轴：采样点 k（0 到 N-1）
- 纵轴：Real part of x(t)
- 标注 PAPR 值（应等于设定值 η=1.1）

### 4.3 图 3(a)：处理增益损失 vs PAPR

**处理增益损失定义**（论文引用 [12], [13]）：
```
loss = 10 * log10( |h^H x|^2 / (||h||^2 * ||x||^2) )
```
即匹配滤波器增益与实际失配滤波器增益的比值（dB），恒为负值。

**仿真**：对 PAPR η ∈ {1, 1.05, 1.1, 1.15, 1.21, 1.44} 分别运行联合设计，计算处理增益损失。

**绘图**：
- 横轴：PAPR（η）
- 纵轴：Loss of processing gain（dB）
- 曲线应随 PAPR 增大而损失减小（更接近 0dB）

### 4.4 图 3(b)：ISLs 和 ILs vs PAPR

**仿真**：同上，对不同 PAPR 计算最终的 ISLs 和 ILs。

**绘图**：
- 横轴：PAPR（η）
- 纵轴：ISLs / ILs（线性或 dB）
- 两条曲线

### 4.5 图 4：干扰环境下脉冲压缩对比（可选）

**场景**：
- 目标在 3000m 处，回波为延迟的发射波形 `x`
- 干扰机在 2000m 处，重发两次 ISRJ 信号，信干比 SJR = -10dB
- 接收信号 = 目标回波 + ISRJ 干扰（两次重发）+ 噪声
- 经过失配滤波器 `h` 后的脉冲压缩输出

**所提方法（图 4(a)）**：使用联合设计的波形 `x` 和滤波器 `h`。
**对比方法（图 4(b)）**：使用 LFM 波形 + 论文 [4] 的方法（参数估计后抑制 ISRJ）。

**绘图**：
- 横轴：距离（m），范围 0 到 6000m
- 纵轴：归一化幅度（dB）
- 标注目标位置（3000m）和假目标位置
- 两种方法使用相同归一化因子

**注意**：对比方法 [4] 的实现较复杂（需参数估计 + 抑制算法），可作为可选扩展。核心需求是图 2 的相关性图像。

---

## 5. 接口规范

### 5.1 主算法接口

```python
def joint_design_algorithm(params: GlobalParams, j: np.ndarray) -> dict:
    """
    交替优化主算法（Algorithm 1）

    输入:
        params: 全局参数
        j: ISRJ采样函数, shape=(N,)

    输出: dict, 包含:
        x: 优化后波形, shape=(N,)
        h: 优化后失配滤波器, shape=(N,)
        isl_history: ISLs收敛历史, list
        il_history: ILs收敛历史, list
        final_isl: 最终ISLs
        final_il: 最终ILs
        processing_gain_loss: 处理增益损失(dB)
    """
```

### 5.2 绘图接口

```python
def plot_correlation_comparison(x: np.ndarray, h: np.ndarray, j: np.ndarray,
                                  params: GlobalParams, save_path: str = None):
    """
    绘制无ISRJ和有ISRJ干扰下的相关性图像（图2）

    输入:
        x: 设计波形
        h: 失配滤波器
        j: ISRJ采样函数
        params: 全局参数
        save_path: 图片保存路径（可选）
    输出: 显示并保存图片
    """

def plot_convergence(isl_history: list, il_history: list, save_path: str = None):
    """绘制目标函数收敛曲线（图1a）"""

def plot_waveform(x: np.ndarray, params: GlobalParams, save_path: str = None):
    """绘制设计波形实部（图1b）"""

def plot_papr_sweep(results: dict, save_path: str = None):
    """绘制处理增益损失和ISLs/ILs vs PAPR（图3）"""

def plot_pulse_compression(received_signal: np.ndarray, h: np.ndarray,
                             params: GlobalParams, method_name: str, save_path: str = None):
    """绘制干扰环境下脉冲压缩结果（图4）"""
```

---

## 6. 验收标准

### 6.1 主算法验收

1. **收敛性**：外层 100 次迭代后，目标函数（ISLs + ILs）收敛到稳定值，最后 10 次迭代变化 < 1%。
2. **约束满足**：
   - `|h^H x - b_max| / b_max < 1e-4`
   - `|h_j^H x - b_min| / |b_min| < 1e-4`
   - `PAPR(x) ≤ η + 0.01`
3. **性能指标**：
   - 处理增益损失 ≈ -0.9dB（论文报告值，允许 ±0.5dB 偏差）
   - ISRJ 信号滤波器输出峰值 ≤ -30dB
   - 波形滤波器输出 PSL（峰值旁瓣电平）< -20dB
4. **内层 ADMM 收敛**：每次外层迭代中，内层 ADMM 的 consensus error（`||x-y||₂ + ||x-z||₂`，两个2范数的和）最终 < 1e-4。

### 6.2 相关性图像验收

1. **无 ISRJ 相关性（图 2a）**：
   - 主瓣在零延迟处（第 N 个采样点），峰值归一化为 0dB
   - 旁瓣电平显著低于主瓣（PSL < -20dB）
   - 处理增益损失标注正确
2. **有 ISRJ 相关性（图 2b）**：
   - 零延迟处峰值 ≤ -30dB（约束要求）
   - 其余延迟处的 ISRJ 输出被深度抑制（< -100dB）
   - 使用与图 2a 相同的归一化因子
3. **双图布局**：1×2 子图，坐标轴标签、标题、图例完整，dB 刻度正确。

### 6.3 其他图表验收

1. **收敛曲线（图 1a）**：ISLs 和 ILs 随迭代单调下降（或总体下降），最终稳定。
2. **波形实部（图 1b）**：波形幅度在 PAPR 约束范围内，PAPR 值标注正确。
3. **PAPR 扫描（图 3）**：处理增益损失随 PAPR 增大而减小（曲线单调上升趋近 0dB）；ISLs 和 ILs 随 PAPR 增大而减小。

---

## 7. 与论文的对应关系

| 论文元素 | 本模块实现 |
|---------|-----------|
| Algorithm 1 | 交替优化主算法 |
| "Initialize x, h, dual variables and auxiliary variables" | 模块一初始化 |
| "Update h using (8) with the fixed x" | 调用模块三（拉格朗日乘子法） |
| "Compute x, y and z using (14), (15) and (16)" | 调用模块四（ADMM） |
| "Update the Lagrangian dual variables using (13d)" | ADMM 对偶变量更新 |
| "end while until convergence" | 内层 consensus error < δ |
| "end for m = M" | 外层固定 100 次 |
| Fig. 1(a) | 目标函数收敛曲线 |
| Fig. 1(b) | 设计波形实部 |
| Fig. 2(a) | 无 ISRJ 相关性（核心需求） |
| Fig. 2(b) | 有 ISRJ 相关性（核心需求） |
| Fig. 3(a) | 处理增益损失 vs PAPR |
| Fig. 3(b) | ISLs 和 ILs vs PAPR |
| Fig. 4 | 干扰环境下脉冲压缩对比（可选） |
| Table I | 仿真参数表（模块一已定义） |

---

## 8. 已确认事项

1. **外层迭代次数 M = 100**：论文明确 M=100。**已确认。**
2. **对偶变量热启动**：论文 Algorithm 1 的初始化在循环外，对偶变量和辅助变量在外层迭代间保持。**已确认使用热启动。**
3. **图 4 对比方法 [4] 暂不实现**：论文 [4] 的方法（参数估计 + ISRJ 抑制）实现复杂。核心需求是图 1-3，图 4 暂不实现。**已确认。**
4. **相关性图像横轴用采样点 k**：图 2 的横轴用 `k`（采样点），不转换为距离。**已确认。**
5. **ISRJ 重发次数与时延**：联合设计阶段（图 1、图 2、图 3）使用单周期无延迟采样函数 `j`；仅图 4 引入两次重发和 2μs 时延。图 4 暂不实现。**已确认。**
6. **码长 N = 800**：论文图 1(b) 横轴到 800 验证。**已确认。**
7. **收敛判据为范数和**：consensus error = `||x-y||₂ + ||x-z||₂ < δ`（两个2范数的和，不是平方和）。**已确认。**
