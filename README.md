# 雷达抗 ISRJ 波形—失配滤波器联合设计

复现 K. Zhou, D. Li, Y. Su, T. Liu, *"Joint Design of Transmit Waveform and
Mismatch Filter in the Presence of Interrupted Sampling Repeater Jamming,"*
IEEE Signal Processing Letters, 2020（DOI: 10.1109/LSP.2020.3021667）。

五个模块与 PRD 文档一一对应，接口与验收标准按 PRD 实现：

| 模块 | 文件 | 对应 PRD |
|------|------|----------|
| 全局参数与 ISRJ 信号生成 | `isrj_joint_design/params.py` | PRD_01 |
| 矩阵构造与目标函数 | `isrj_joint_design/matrices.py` | PRD_02 |
| 拉格朗日乘子法求解滤波器 | `isrj_joint_design/filter_solver.py` | PRD_03 |
| ADMM 求解波形 | `isrj_joint_design/waveform_admm.py` | PRD_04 |
| 主算法与相关性绘图 | `isrj_joint_design/joint_design.py`、`plotting.py` | PRD_05 |

## 文档体系

| 文档 | 内容 |
|------|------|
| [CONTEXT.md](CONTEXT.md) | 术语表（本项目语境下的专有概念口径） |
| PRD_01 ~ PRD_05 | 需求规格（唯一权威，含"已确认事项"裁决） |
| [CLAUDE.md](CLAUDE.md) | 工作约定 + 关键实现决策 + 已知偏差 |
| [docs/过程文档.md](docs/过程文档.md) | 全过程记录：时间线、决策日志、实验记录 |
| [docs/知识库.md](docs/知识库.md) | 推导成果与实验结论库 |
| [docs/adr/](docs/adr/) | 4 条关键决策（共轭约定、式(8) 勘误、损失验收口径、默认窗口） |
| [docs/验收对照表.md](docs/验收对照表.md) | 每个 PRD 验收项的实测数值与结论 |

## 环境与运行

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt     # Windows
source .venv/bin/activate && pip install -r requirements.txt   # Linux/macOS
```

复现论文图 1、图 2（论文设定 N=800、M=100，约 1 分钟）：

```bash
.venv/Scripts/python.exe scripts/run_joint_design.py
```

> 默认采样窗口为 PRD 基线的 [0, 2.5) μs。实验旋钮：`--tau-offset`（窗口起点）、
> `--tau`（窗口宽度）等，见 `--help`。

复现论文图 3 的 PAPR 扫描（6 个点，可加 `--repeats 4` 做多起点；用
`--init lfm` 可得到论文报告的处理增益损失水平，见下）：

```bash
.venv/Scripts/python.exe scripts/run_papr_sweep.py
.venv/Scripts/python.exe scripts/run_papr_sweep.py --init lfm --repeats 4
```

图像输出到 `figures_初始0/`，数值结果保存为 `figures_初始0/joint_design_result.npz` 与
`figures_初始0/papr_sweep.json`。

## 测试

```bash
.venv/Scripts/python.exe -m pytest tests/ -q
```

每个测试文件对应一个 PRD 的验收标准（PRD_01 §6 → `test_params.py`，依此类推）。

## 关键实现决策（详见 CLAUDE.md）

- **矩阵共轭约定**：卷积/互相关矩阵由共轭序列铺成（`H[k,n] = conj(h[k-N+n])`），
  使主瓣峰值严格等于 PRD 中的 `h^H x`；此时 `H x = conj(X h)`，目标函数两种
  表达严格相等。
- **式 (8) 勘误已修正**：滤波器闭式解采用"两项含共轭"形式（论文印刷版遗漏
  `Q^{-1}x_j` 项且第二项系数漏共轭）。
- **`P`/`Q` 用 Toeplitz 结构快速组装**：`H^H D H = Toeplitz(r_h) - h h^H`，
  `H_j^H H_j = diag(j) Toeplitz(r_h) diag(j)`，无需构造 `(2N-1)×N` 大矩阵。
- **处理增益损失对初始化敏感**：PRD 指定的随机相位初始化下损失 −2.5 dB
  （论文 −0.9 dB），换来 PSL −31.6 dB；`init="lfm"` 下损失 −1.32 → −0.50 dB
  随 PAPR 单调改善（η=1.1 时 −0.84 dB，与论文 −0.9 dB 吻合），代价是 PSL
  差约 5–10 dB。详见 CLAUDE.md "已知偏差"。两种初始化的扫描结果分别保存在
  `figures_初始0/papr_sweep_lfm_r4.json` 与 `figures_初始0/papr_sweep_random_phase_r4.json`。
