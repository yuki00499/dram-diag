# DRAM 晶圆缺陷诊断：可靠长尾多标签识别

基于 1148 张人工复核 SEM 图像的 7 类多标签缺陷属性识别项目。当前主协议为 `dram-ml-v2`，以核心标签 `2/3/4/5` 的 Macro-F1 为主指标，标签 `1/7/8` 作为少样本探索结果单独报告。

## 1. 环境与测试

要求 Python 3.12：

```powershell
Set-Location E:\Ai\dram-diag
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -e . --no-build-isolation
python -m pytest -q
```

## 2. 冻结数据协议

```powershell
python scripts\convert_labels.py --labels label_v2.json --out artifacts\dram_ml_v2
python scripts\audit_dataset.py `
  --protocol configs\protocols\dram_ml_v2.yaml `
  --labels artifacts\dram_ml_v2\labels.csv `
  --source-labels label_v2.json `
  --out artifacts\dram_ml_v2
```

协议锁定 115 张最终测试图，其余 1033 张组成 development，并生成 5 个迭代多标签分层折。审计同时输出标签支持度、组合频率、共现矩阵、每折覆盖、重复图像、低质量图、标注哈希和代码版本。

## 3. 正式训练

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

三种模型分别是：

- `bce`：普通 ResNet18 + BCE。
- `weighted_bce`：按训练折统计正样本，使用平方根缩放且上限为 5 的 `pos_weight`。
- `graph_bce`：在长尾加权模型上加入训练折共现图残差，只保留支持数不少于 5 的关系。

## 4. 模型选择与锁定测试

```powershell
python scripts\select_multilabel_model.py
python scripts\evaluate_multilabel_test.py
python scripts\render_competition_report.py
```

选择工具只在共现图模型的核心 Macro-F1 不低于长尾基线，且核心或共现组合 F1 至少提升 0.005 时保留创新模型，否则自动回退到 `weighted_bce`。

`evaluate_multilabel_test.py` 会验证 checkpoint 哈希，并拒绝覆盖已有测试报告。只有模型结构、阈值和集成规则完全锁定后才能执行；该命令会正式消费最终测试集，不应用于调参。

报告渲染器会生成标签共现热力图、三模型 OOF 对比图和最终测试 PR 曲线；测试报告同时保存典型成功与失败案例清单。

## 5. 部署前端服务

```powershell
python scripts\deploy_multilabel.py `
  --selection artifacts\dram_ml_v2\model_selection.json `
  --out artifacts\deployment-v2.json
python scripts\serve.py --port 8010
```

服务启动在 `http://127.0.0.1:8010`（默认端口已改为 8010，默认加载 `artifacts\deployment-v2.json`）。也可用环境变量方式启动：

```powershell
$env:DRAM_DEPLOYMENT = "artifacts/deployment-v2.json"
python scripts\serve.py --port 8010
```

前端为四页单页应用（`web/`，Modern Minimalist 设计）：

- **数据概览**：协议信息、缺陷族分布、数据划分与缺陷族清单
- **模型评估**：OOF 指标（Macro-F1、Sample-F1、mAP、Exact Match、Hamming）、训练曲线、逐类型混淆统计、PR 曲线、组合指标、折间稳定性与阈值稳定性
- **单图诊断**：数据集选图（按集合浏览/搜索）或上传 SEM 图像 → 逐标签阈值检出、状态徽章（可信/置信偏低/无检出/质量不足）、真实标签对照、相似案例
- **批量诊断**：数据集抽样（服务端种子抽样）或批量上传 → 结果表（预测 vs 真实标签对照）、筛选（类型/需复核/文件名）、CSV 导出

部署清单支持 5 折集成与逐标签阈值，模型接口额外返回 `thresholds` 与 `review_margin`。

详细设计见 `项目方案/PLAN.md`。正式 v2 模型训练完成后，将旧的运行时 fallback 产物替换为模型选择清单指定的5折模型。
