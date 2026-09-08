# DRAM 晶圆缺陷诊断：可靠长尾多标签识别

基于 1148 张人工复核 SEM 图像的 7 类多标签缺陷属性识别项目。一张晶圆 SEM 图像可以同时包含多种缺陷（如「细长颗粒」与「划痕」共存），因此任务被建模为多标签分类而非单标签分类。

项目以**可靠长尾多标签学习**为核心，重点解决类别极不均衡、缺陷共现、阈值不稳定和测试泄漏四个问题：

- 冻结人工标注源 `label_v2.json`，任何修改都必须改变数据指纹并提升实验版本。
- 模型统一输出 7 个独立缺陷概率，允许单图同时检出多个缺陷。
- 通过「普通 BCE → 长尾加权 BCE → 共现图残差模型」的固定对照证明创新收益；复杂模型无稳定收益时自动回退。
- 主结论仅针对样本较充分的核心标签 `2/3/4/5`（Macro-F1 主指标）；稀有标签 `1/7/8` 作为少样本探索结果单独报告。
- 测试集严格锁定为 115 张，只允许最终评估一次，防止测试泄漏。

当前主协议为 `dram-ml-v2`。

## 1. 数据集

### 1.1 数据规模

- 原始图像：1150 张，480×320 灰度 SEM 图像。
- 有效标注：1148 张（排除 `image_824.jpg`、`image_895.jpg`，原因均为未完成标注）。
- 单标签图像 766 张、双标签图像 381 张、三标签图像 1 张。

### 1.2 缺陷类型定义与支持度

| ID | 缺陷属性（英文） | 中文含义 | 样本数 | 评估角色 |
|---:|---|---|---:|---|
| 1 | Round Particle | 圆形颗粒、异物 | 18 | 稀有标签，探索性报告 |
| 2 | Elongated Particle | 细长颗粒、异物 | 681 | 核心标签 |
| 3 | Polygonal Particle | 方形颗粒、异物 | 160 | 核心标签 |
| 4 | Scratch / Crack | 划痕、裂纹 | 552 | 核心标签 |
| 5 | Pit / Void | 凹坑、空洞 | 87 | 核心标签 |
| 7 | Blob | 块状、块斑类 | 14 | 稀有标签，探索性报告 |
| 8 | Low-signal | 背景、低信号 | 19 | 稀有标签，探索性报告 |

标签 6（Dark spot）与 9（Unknown）未进入训练：6 未在最终标签体系中使用，9 仅作为标注过程中的 Unknown 操作项。

### 1.3 重点共现关系

最稳定、且具有诊断意义的共现组合（仅作为模型辅助证据，不解释为工艺因果）：

| 组合 | 共现数 |
|---|---|
| `2+4`（细长颗粒 + 划痕） | 251 |
| `3+4`（方形颗粒 + 划痕） | 40 |
| `4+5`（划痕 + 凹坑） | 82 |

## 2. 方法

### 2.1 通用训练设置

- Backbone：ImageNet 预训练 ResNet18。
- 输入：480×320 等比例 letterbox，灰度复制为三通道，归一化只使用当前训练折计算的 mean/std。
- 增强：轻度旋转、亮度、对比度和轻微模糊；不使用会破坏小缺陷的强裁剪或擦除。
- 前 5 个 epoch 冻结编码器，之后解冻微调；编码器学习率 `1e-4`、分类头 `3e-4`，AdamW + cosine 衰减。
- 最多 60 个 epoch，patience 12，按核心标签 Macro-F1 保存最佳 checkpoint；正式训练固定 seed 42。

### 2.2 三个候选模型（固定对照）

| 配置 | 说明 |
|---|---|
| `dram_ml_v2_bce.yaml` | 普通 ResNet18 + 标准 `BCEWithLogitsLoss`，不引入类别权重或标签关系，作为基础基线 |
| `dram_ml_v2_weighted.yaml` | 长尾加权 BCE：`pos_weight = clamp(sqrt(N_neg/N_pos), 1, 5)`，平方根缩放降低长尾漏检、上限 5 防止稀有标签主导梯度；权重只按训练折统计 |
| `dram_ml_v2_graph.yaml` | 共现图残差模型（主要创新）：仅保留训练支持数 ≥ 5 的共现关系构建标签图，`logits = raw + sigmoid(alpha) * tanh(raw) @ label_graph`，先依赖图像证据、再学习共现修正；标签图随 checkpoint 保存 |

### 2.3 阈值校准与集成

- 固定 0.5 仅用于训练过程观察；所有 development 样本获得 OOF 概率后，对每个标签在 `[0.05, 0.95]` 以 0.01 步长搜索 F1 最优阈值。
- 核心标签使用 OOF 最优阈值；稀有标签执行 500 次分层 bootstrap 取中位数，并按 20 个先验样本向 0.5 收缩。
- 测试集与竞赛待预测数据绝不参与权重、图结构或阈值选择。
- 最终集成：同一候选的 5 个折模型概率算术平均；平均概率距任一阈值不超过 0.1 时标记 `low_confidence`，供演示系统提示复核。

### 2.4 模型保留与回退规则

共现图模型只有同时满足以下条件才被保留：

- 核心标签 OOF Macro-F1 不低于加权 BCE；
- 核心 Macro-F1 或三组共现组合平均 F1 至少提升 0.005。

否则自动回退至 `weighted_bce`；普通 BCE 只作为基础对照。

## 3. 评估体系

- **主指标**：核心标签 `2/3/4/5` 的 Macro-F1（避免高频标签掩盖标签 3/5，也避免极少样本标签决定结论）。
- **辅助指标**：全 7 标签 Macro-F1、Micro-F1、样本级 F1、mAP、逐标签 PR-AUC / P / R / F1 / 支持度、Exact Match、Hamming Loss、5 折均值/标准差/最低折。
- **共现组合指标**：分别报告 `2+4`、`3+4`、`4+5` 的 Precision / Recall / F1，证明创新收益来自正确识别共现缺陷而非仅调整阈值。
- **可靠性报告**：核心 Macro-F1 的 2000 次分层 bootstrap 95% 置信区间、阈值 bootstrap 区间、逐标签 PR 曲线、典型成功与失败案例；稀有标签单独列出并明确标注为探索性结果。
- **测试集锁定**：只有模型结构、阈值和集成规则完全锁定后才执行正式测试评估；`test_report.json` 存在时拒绝覆盖，测试集只消费一次。

## 4. 环境与测试

要求 Python 3.12：

```powershell
Set-Location E:\Ai\dram-diag
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -e . --no-build-isolation
python -m pytest -q
```

测试覆盖数据/协议校验、指标与检索、旧模型兼容与协议冻结（`tests/`）。

## 5. 冻结数据协议

```powershell
python scripts\convert_labels.py --labels label_v2.json --out artifacts\dram_ml_v2
python scripts\audit_dataset.py `
  --protocol configs\protocols\dram_ml_v2.yaml `
  --labels artifacts\dram_ml_v2\labels.csv `
  --source-labels label_v2.json `
  --out artifacts\dram_ml_v2
```

协议锁定 115 张最终测试图，其余 1033 张组成 development，并生成 5 个迭代多标签分层折（每张 development 图像恰好作为一次验证样本，训练/验证/测试互不重叠）。审计同时输出标签支持度、组合频率、共现矩阵、每折覆盖、重复图像、低质量图、标注哈希和代码版本。

## 6. 正式训练

按固定顺序运行三个候选。每个配置默认训练 5 折并生成 OOF 概率、逐标签阈值、阈值 bootstrap、折间波动和核心 Macro-F1 置信区间。

```powershell
python scripts\train.py --config configs\experiments\dram_ml_v2_bce.yaml
python scripts\train.py --config configs\experiments\dram_ml_v2_weighted.yaml
python scripts\train.py --config configs\experiments\dram_ml_v2_graph.yaml
```

快速检查共现图链路：

```powershell
python scripts\train.py `
  --config configs\experiments\dram_ml_v2_graph.yaml `
  --epochs 1 --folds 0 --seeds 42 `
  --out-dir runs\smoke\dram_ml_v2_graph
```

## 7. 模型选择与锁定测试

```powershell
python scripts\select_multilabel_model.py
python scripts\evaluate_multilabel_test.py
python scripts\render_competition_report.py
```

- `select_multilabel_model.py`：按预设规则选择共现图模型或回退至 `weighted_bce`，生成冻结协议指纹、checkpoint 哈希、阈值与模型版本的 `model_selection.json`。
- `evaluate_multilabel_test.py`：验证 checkpoint 哈希、拒绝覆盖已有测试报告；该命令会正式消费最终测试集，不应用于调参。
- `render_competition_report.py`：生成标签共现热力图、三模型 OOF 对比图、最终测试 PR 曲线与典型成败案例清单。

## 8. 部署前端服务

```powershell
python scripts\deploy_multilabel.py `
  --selection artifacts\dram_ml_v2\model_selection.json `
  --out artifacts\deployment-v2.json
python scripts\serve.py --port 8010
```

服务启动在 `http://127.0.0.1:8010`（默认加载 `artifacts\deployment-v2.json`）。也可用环境变量方式启动：

```powershell
$env:DRAM_DEPLOYMENT = "artifacts/deployment-v2.json"
python scripts\serve.py --port 8010
```

部署清单支持 5 折集成与逐标签阈值，模型接口额外返回 `thresholds` 与 `review_margin`。

### 8.1 REST API

| 端点 | 说明 |
|---|---|
| `GET /api/model/info` | 模型与协议信息 |
| `GET /api/model/history` | 训练历史 |
| `GET /api/model/class-distribution` | 类别分布 |
| `GET /api/model/evaluation` | 评估结果 |
| `GET /api/dataset/images` | 数据集图片列表（按集合浏览/搜索） |
| `GET /api/dataset/image/{image_name}` | 单张图片信息与真实标签 |
| `POST /api/diagnose/upload` | 单图上传诊断 |
| `POST /api/diagnose/batch` | 按图片名批量诊断 |
| `POST /api/diagnose/batch-upload` | 批量上传诊断 |

### 8.2 前端（`web/`，四页单页应用，Modern Minimalist 设计）

- **数据概览**：协议信息、缺陷族分布、数据划分与缺陷族清单
- **模型评估**：OOF 指标（Macro-F1、Sample-F1、mAP、Exact Match、Hamming）、训练曲线、逐类型混淆统计、PR 曲线、组合指标、折间稳定性与阈值稳定性
- **单图诊断**：数据集选图或上传 SEM 图像 → 逐标签阈值检出、状态徽章（可信/置信偏低/无检出/质量不足）、真实标签对照、相似案例检索
- **批量诊断**：数据集抽样（服务端种子抽样）或批量上传 → 结果表（预测 vs 真实标签）、筛选（类型/需复核/文件名）、CSV 导出

## 9. 项目结构

```
dram-diag/
├── label_v2.json              # 冻结的人工复核标注源（数据指纹）
├── configs/
│   ├── protocols/dram_ml_v2.yaml   # 冻结数据协议（类型、划分、折数、种子）
│   └── experiments/                # 三个候选模型配置
├── scripts/                    # 训练/审计/选择/评估/部署/服务 命令行入口
├── src/dram_diag/              # 核心库（协议、数据、模型、训练、推理、指标、检索、API）
├── web/                        # 四页 SPA 前端（index.html / app.js / charts.js / styles.css）
├── tests/                      # pytest 测试（协议、数据、指标、兼容）
├── artifacts/                  # 生成产物（manifest、选择清单、部署清单、报告）
├── runs/                       # 训练输出（checkpoints、OOF、报告）
├── 晶圆缺陷分类数据集/          # 原始 SEM 图像
├── requirements.txt
└── pyproject.toml
```

## 10. 参考

正式 v2 模型训练完成后，将旧的运行时 fallback 产物替换为模型选择清单指定的 5 折模型。
