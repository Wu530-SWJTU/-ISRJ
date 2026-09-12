## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

## 项目背景

本项目复现论文 *Joint Design of Transmit Waveform and Mismatch Filter in the
Presence of Interrupted Sampling Repeater Jamming*（Kai Zhou, Dexin Li, Yi Su,
Tao Liu, IEEE SPL 2020, DOI: 10.1109/LSP.2020.3021667）：交替优化发射波形 `x`
与失配滤波器 `h`，在 ISRJ（中断采样转发干扰）下同时压低脉冲压缩旁瓣（ISLs）
与干扰输出电平（ILs）。

实现完全由 PRD 文档集驱动。**PRD 是唯一权威**：五个 PRD（PRD_01 ~ PRD_05）
位于仓库根目录，每个模块对应一个 PRD，函数名、公式、参数取值、验收标准均以
PRD 为准。论文本身存在排版错误与含糊之处，PRD 的"已确认事项"小节是裁决结果
（见下文"关键实现决策"）。

**"PRD 唯一权威"仅适用于复现线**（`isrj_joint_design/`、`tests/` 与三个复现
脚本，见下表）。仓库另有一条**搜索线**（ISRJ 参数搜索），权威文件与口径不同，
见下文"研究线：参数搜索"一节。

### 模块与 PRD 对应关系

| 模块 | 文件 | PRD | 论文对应 |
|------|------|-----|----------|
| 全局参数与 ISRJ 信号生成 | `isrj_joint_design/params.py` | PRD_01 | Table I / Section IV |
| 矩阵构造与目标函数 | `isrj_joint_design/matrices.py` | PRD_02 | Section II 式 (1)–(6) |
| 拉格朗日乘子法求解滤波器 | `isrj_joint_design/filter_solver.py` | PRD_03 | Section III 式 (7)–(8) |
| ADMM 求解波形 | `isrj_joint_design/waveform_admm.py` | PRD_04 | Section III 式 (9)–(16) |
| 主算法与指标 | `isrj_joint_design/joint_design.py` | PRD_05 | Algorithm 1 |
| 绘图（图 1–3） | `isrj_joint_design/plotting.py` | PRD_05 §3–§4 | Fig. 1–3 |
| 复现脚本 | `scripts/run_joint_design.py`、`scripts/run_papr_sweep.py` | PRD_05 §5 | — |
| 验收测试 | `tests/test_<模块>.py`，每文件对应一个 PRD 的验收标准 | — | — |

图 4（干扰环境脉冲压缩对比）按 PRD_05 §8.3 **已确认不实现**。

## 研究线：参数搜索（分支 `参数搜索`）

复现线之外的另一条研究线：把联合设计当作**代价函数构造工具**，反向搜索 ISRJ
参数（当前一维瞄准采样周期），使其逼近真实参数。

- **权威文档**：`docs/参数搜索/搜索线规格说明.md`（口径 + 工程约定，**开发时以此为准**）、
  `docs/参数搜索/组会汇报.md`（方案全文，含符号对照表与 WJPS 矩阵链）、
  `docs/参数搜索/工作流.md`（W0–W6 阶段卡片、门禁、算力预算）。PRD **不适用**于搜索线。
- **复现线冻结**：`isrj_joint_design/` 与 `tests/` 只接受"为通过 PRD 验收"的改动。
  搜索线代码放 `isrj_search/`（W1 开工时创建）、脚本放 `scripts/run_search_*.py`，
  只调用复现线的公开接口（清单见规格说明 §1.1），不修改它。
- **参数向量**：`θ = [t_0, d, T_s]`（转发起始时间 / 占空比 / 采样重复周期），
  `T_L = d·T_s` 为派生量（截取长度，不进搜索向量）。
- **代价函数**：`C(θ̂) = max|y_j(θ*) ∗ h(θ̂)| / max|x(θ̂) ∗ h(θ̂)|`，
  **分子必须用真实参数 θ\* 生成的干扰**；分母 = `b_max = N`，由硬约束钉死。
  分子若写成候选参数 `j(θ̂)`，C 会退化为常数 `b_min/b_max` 而与 θ̂ 无关，搜索失效
  （证明见组会汇报 §2.3）。等价指标：`C = 10^(isrj_peak_db/20)`。
- **干扰模型**：转发时延 0、边采样边转发、单份原位，`y_j = x ⊙ j`；
  多份转发与论文图 4 的干扰机时延不建模。
- **分支策略**：搜索线占一条长期分支（W1–W6 依次依赖），**不按实验族拆分支**。
- **5 个决策门**（规格说明 §3）未拍板前不得实现对应阶段，尤其 WJPS 的权重与约束形式。

## 用户工作约定

- **每次改动干扰信号模型（采样窗口、时延、重发次数/偏移、转发方式等），必须在
  回复中说明新的 `H_j` / `X_j` 矩阵构造**：给出信号级公式（如
  `y_jam = Σ_r shift_{d_r}(x⊙j)`）与矩阵链（如 `J_multi = (I+S_d)·diag(j)`），
  不能只给代码。用户 2026-08-30 约定。

## 环境与运行

- 虚拟环境 `.venv/`（Python 3.13），依赖见 `requirements.txt`：
  numpy + scipy（Cholesky/Toeplitz）+ matplotlib + pytest。无 GPU 需求。
- Windows 下运行：`.venv/Scripts/python.exe ...`（**不要**用裸 `python`，全局
  解释器没有装依赖）。
- 跑完整仿真：`.venv/Scripts/python.exe scripts/run_joint_design.py`
  （N=800、M=100，约 1 分钟）；PAPR 扫描：`scripts/run_papr_sweep.py`。
- 测试：`.venv/Scripts/python.exe -m pytest tests/ -q`。**改代码后必须先跑
  全量测试**；测试按 PRD 验收标准编写，是验收的裁判。

## 关键实现决策（变更前必读）

以下决策经过了推导与数值验证，改动任何一条都会破坏与 PRD/论文的对应关系：

1. **矩阵共轭约定**（`matrices.py` 顶部 docstring 有完整论证）：卷积/互相关
   矩阵由**共轭序列**铺成 —— `H[k,n] = conj(h[k-N+n])`、
   `X[k,n] = conj(x[N+n-k])`。这样脉压输出的第 N 个元素严格等于 PRD 中的
   `h^H x`/`x^H h`，且 `x^H P x == h^H Q h` 严格成立。代价是 `H x = conj(X h)`
   （模长一致），PRD_02 验收 1 的"Hx 与 Xh 数值一致"在本实现中解释为
   "共轭一致"——`H x` 对 `h` 共轭线性，不可能同时等于对 `h` 线性的 `X h`，
   二者只能取一，取"带共轭"才与 PRD 全部其余公式自洽。
2. **式 (8) 勘误已修正**（PRD_03 §3.4）：滤波器闭式解是 `Q^{-1}x` 与
   `Q^{-1}x_j` 的**两项**线性组合，第二项系数含共轭 `conj(b)`。论文印刷版遗漏
   第二项且漏共轭。测试用独立 KKT 线性方程组解法交叉验证了本实现。
3. **`P`/`Q` 快速组装**（PRD_02 §4.3）：`H^H D H = Toeplitz(r_h) - h h^H`、
   `H_j^H H_j = diag(j)·Toeplitz(r_h)·diag(j)`、`X_j^H X_j` 是 `x⊙j` 的自相关
   Toeplitz 阵（**不是** `diag(j) X^H X diag(j)`）。FFT 算自相关，`scipy.linalg.
   toeplitz` 铺矩阵，O(N² log N)。显式构造函数仍保留在 `matrices.py` 中，
   测试断言两条路径数值等价 —— 改动快速路径时这个测试会抓住回归。
4. **ADMM 系数必须带 `2P`**（PRD_04 §10.2）：`A = 2P + 2ρI + ρhh^H + ρh_jh_j^H`
   （复变量拆实虚部求导的结果），用 P 是错的。
5. **收敛判据是范数和**（PRD_04 §10.3）：`||x-y||₂ + ||x-z||₂ < δ`，不是
   平方和。`δ = 1e-4`、`max_inner_iter = 500`、`ρ = 5000` 均已确认。
6. **对偶变量初始化**：PRD_01 §4.4（论文原文）要求随机相位初始化，
   PRD_05 §2.1 伪代码写零初始化。默认按论文用随机相位，`dual_init="zeros"`
   可切换。两种初始化都通过了内层收敛测试。
7. **热启动**（PRD_05 §2.2）：`y, z, λ, γ, μ, ν` 在外层迭代之间保持，
   不重新初始化。测试 `test_warm_start_state_is_carried_across_outer_iterations`
   断言内层迭代次数随外层迭代下降（实测 500 → ~35）。
8. **ADMM 只接可行的 h**：随机单位模 `h` 下约束集为空
   （`|h^H x| ≤ √N·√N = N` 取等要求 `x = h`，此时 `h_j^H x` 恒为 j 的非零个数
   ≠ b_min），ADMM 无法收敛。真实调用顺序必须先解 h（模块三）再进 ADMM
   （模块四），`tests/test_waveform_admm.py` 的 fixture 即按此构造。
9. **内层 ADMM 的预热**：外层前 1–2 轮 consensus error 可能停在 ~1e-3（触顶
   500 次迭代），随后热启动生效、每轮收敛到 < 1e-4。全规模实测只有第 1 轮
   如此，小规模 fixture 是前 2 轮 —— 不是 bug。
10. **波形初始化**（`params.init`）：默认 `"random_phase"`（PRD_01 §4.1 指定）。
    `"lfm"` 用单位模线性调频（`exp(jπBt²/T_p)`，fs=B）初始化 —— 实测这会让
    处理增益损失降到论文报告的 −0.9dB 附近并呈现图 3(a) 的单调趋势，代价是
    PSL 变差约 8dB（见"已知偏差"）。
11. **采样窗口默认位置**（`params.tau_offset`）：**默认 `0.0`，与 PRD_01 §3.2
    一致（窗口 [0, 2.5) μs）**。历史：曾按用户实验要求默认 `2.5e-6`（窗口
    [2.5, 5) μs，见 ADR-0004），2026-08-30 恢复 PRD 基线。实测纯平移对指标
    无统计影响（损失 −2.49 → −2.51 dB、PSL −31.6 → −31.9 dB，在种子波动内），
    但内层 ADMM 迭代变慢、总耗时约翻倍（h_j 支撑移到中间带导致 A 的条件数
    变差）。

## 已知偏差（如实汇报，勿"修"掉）

**处理增益损失对波形初始化敏感（2026-08-25 全规模 N=800/M=100 实测）：**

- 默认随机相位初始化（PRD_01 指定）：损失 −2.5 dB（论文 −0.9 dB），不同种子
  极差仅 0.3 dB，第 1 次外层迭代即定型，**与收敛无关**；PSL −31.6 dB。
- 最小范数滤波器（只满足两个峰值约束）在收敛波形上的损失为 −0.77 dB，随
  PAPR 单调改善（η=1.0 → 1.44 时 −0.98 → −0.14 dB）—— 这是任何带 ISLs/ILs
  目标的解的**理论上界**，且与论文趋势一致；但它的 PSL 仅 −14.5 dB、ISRJ
  峰值 −26.7 dB（不达标）。
- `init="lfm"`（多起点 ×4，2026-08-25 实测）：损失 **−1.32 → −0.50 dB 随
  PAPR 单调改善**（η=1.1 时 −0.84 dB，论文 −0.9 dB；η=1.44 时 −0.50 dB），
  完全复现论文图 3(a) 的趋势；起点间损失极差仅 0.01–0.08 dB（随机相位时为
  0.3–0.4 dB，LFM 初值下优化器确定性更强）。PSL −21 到 −27 dB（仍满足
  < −20 dB 验收），比随机相位的 −31.6 dB 差。ISLs 总体下降但 η=1.15→1.44
  段有轻微回弹（1.65e4→1.76e4，属于验收容许的"总体下降"）。说明论文报告的
  −0.9 dB 对应的是**低旁瓣要求较松的局部最优**，初始化决定了落在哪个盆。
- 图 3(a) 的趋势验收（PRD_05 §6.3）只在 `init="lfm"` 下成立；随机相位初始化
  时损失在 −2.5±0.3 dB 内无单调性（波动大于 PAPR 差异）。图 3(b) 的 ISLs/ILs
  总体下降两种初始化都成立。
- 复现数据文件（`figures_初始0/` 下）：`papr_sweep_random_phase_r4.json`、
  `papr_sweep_lfm_r4.json`（LFM 另含全部起点的 `papr_sweep_all_runs_lfm_r4.json`），
  文件名带 `init` 与 `repeats` 标记，互不覆盖。
- 其余验收指标全部达标：主瓣/干扰约束残差 < 1e-7、PAPR 精确钉在 η、
  内层 ADMM 每轮收敛到 < 1e-4、PSL < −20 dB、ISRJ 零延迟 −30.00 dB。

若将来需要同时逼近 −0.9 dB 与 −30 dB 旁瓣：优先怀疑 (a) 论文对损失或 ISLs
的定义细节（如主瓣屏蔽区宽度、归一化）；(b) ε 的实际取值；(c) 滤波器长度
> N 的可能性；(d) 外层迭代次数或外层收敛判据。**不要**为了凑数添加 PRD 未
规定的正则化旋钮，除非用户明确要求。

## 备选后续任务（用户已确认暂缓）

深挖"论文为何同时报告 −0.9 dB 与低旁瓣"的专项调查，**用户 2026-08-25 拍板暂缓**。
若写报告/论文需要给审稿人交代时启动，按成本顺序：

1. **(a) 主瓣屏蔽区宽度**（约 1 小时）：`D` 目前只屏蔽主瓣 1 点；试屏蔽 ±1~±3
   点，看 \|h\|²（即损失）是否下降。归一化项可推理排除（常数缩放不改最优解）。
2. **(b) ε 实际取值**（约 1 小时）：快跑（M=25）探测已显示 ε=0.1 → 损失 −1.95 dB
   （PSL −22.8），ε 是真实杠杆。扫 ε∈{0, 0.1, 0.2, 0.3, 0.4, 0.5} 看是否逼近 −0.9。
3. **(c) 滤波器长度 > N**（约半天，代价最高）：可干净解释"低损失+低旁瓣"，
   但违背 PRD 的 h∈C^N 维度假设（需改模块二/三矩阵构造与两个求解器）；且最小
   范数下界不随长度改善（超长部分能量最优解为零），非稳赢。
4. **(d) 外层收敛判据**：**已排除**（损失第 1 轮即定型，M=25 与 M=100 无差别）。

## 参数口径速查

- 码长 `N = 800`（复采样 `fs = B = 20MHz`，`T_p = 40μs`），`j` 的周期 200 点、
  脉宽 50 点（占空比 0.25），默认窗口 [0, 2.5) μs（PRD 基线，决策 #11）。
  实验旋钮：`tau_offset`（窗口起点）、`retransmit_offsets`（评估侧多转发偏移，
  默认 (0.0,) 单份零延迟）。
- `b_max = N`、`b_min = N·10^(-1.5)`（即 20lg(b_min/b_max) = −30dB）。
- 论文图 4 才用到的干扰机时延（2μs）/两次重发在联合设计阶段**不建模**。
- 跑非论文规模（N≠800）时用 `GlobalParams(N=..., T_p=N/20e6, ...)` 保持
  `N = B·T_p` 自洽（`run_joint_design.py` 已内置此换算）。

## 测试与验收

- 每个测试文件对应一个 PRD 的验收标准小节；测试内 docstring 标注
  "PRD_xx §y 验收 z"。新增/修改功能时对照对应 PRD 更新测试。
- 数值类断言统一用 `pytest.approx`/`np.allclose` 及 PRD 给定的容差
  （如约束残差 1e-6、Hermitian 性 1e-10）。
- `figures_初始0/` 与 `.venv/` 已 gitignore；`figures_初始0/*.npz`/`*.json` 是复现运行的
  数值记录，删除无害（脚本可重新生成）。
