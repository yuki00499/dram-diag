# DRAM 晶圆缺陷诊断（多标签版）

基于缺陷复查 SEM 图像的**多标签缺陷族分类**项目。原始数据标签经人工复核确认全部错误后，已通过自建标注站重新打标（7 个缺陷族，每图可多选），并重建了完整的多标签训练/评估/部署链路。

## 当前状态

- 协议 `ge20-ml-v1`：7 个缺陷族（1 圆形颗粒/异物、2 细长颗粒/异物、3 方形颗粒/异物、4 划痕/裂纹、5 凹坑/空洞、7 块状/块斑类、8 背景/低信号）
- 数据集：1150 张 480×320 灰度 SEM 复查图，其中 1148 张已标注（2 张未标已排除）
- 划分：train 918 / validation 115 / test_known 115（按图片随机，无类别互斥）
- 基线（ResNet18 + dataset_gray + BCE）：**验证 Macro-F1 0.763、Exact-match 83.5%**

## 1. 环境准备

要求 Python 3.12。若现有 `.venv` 可用：

```powershell
Set-Location E:\Ai\dram-diag
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -r requirements.txt
python -m pip install -e . --no-build-isolation
python -m pytest -q
```

## 2. 重新打标（需要人工时）

```powershell
python scripts\build_multi_label_app.py --detach
```

- 单图顺序流（image_0 → image_1149），键盘 `1-9` 勾选缺陷族，`Enter`/`→` 下一张
- 8（背景/低信号）与 9（Unknown）互斥，勾选自动清空其他
- 进度自动保存（localStorage）；完成后点"导出标注"得到 `label_v2.json`
- 停止服务：`python scripts\stop_labeling.py`

## 3. 标注转换与协议审计

将标注导出转换为多标签 CSV 并生成冻结 manifest：

```powershell
python scripts\convert_labels.py --labels label_v2.json --out artifacts\ge20_ml
python scripts\audit_dataset.py --data-root 晶圆缺陷分类数据集 --protocol configs\protocols\ge20_ml.yaml --labels artifacts\ge20_ml\labels.csv --out artifacts\ge20_ml
```

- `convert_labels.py` 会删除未使用的类型、排除未标图片，并输出统计报告（`convert_report.json`）
- `audit_dataset.py` 校验图片完整性与标签合法性，生成 `split_manifest.json`（含指纹，防篡改）

## 4. 训练

```powershell
# 冒烟测试（1 epoch）
python scripts\train.py --config configs\experiments\multilabel_baseline.yaml --epochs 1 --seeds 42 --out-dir runs\smoke\multilabel

# 正式基线
python scripts\train.py --config configs\experiments\multilabel_baseline.yaml --seeds 42
```

- 模型：ResNet18 预训练 + 冻结 5 epoch 后解冻微调，`BCEWithLogitsLoss` 多标签
- 归一化：`dataset_gray`（灰度数据自身统计）
- 选模指标：validation Macro-F1（per-label F1/AUC、exact-match 同步记录在 history）
- 多 seed：`--seeds 42 43 44`

## 5. 部署到 Web 前端

```powershell
python scripts\deploy_multilabel.py --checkpoint runs\ge20_ml_baseline\seed-42\best.pt
python scripts\serve.py
```

浏览器访问 `http://127.0.0.1:8000`：

- **概览**：7 缺陷族分布、训练曲线、验证指标
- **单图诊断**：上传图片 → 各缺陷族概率条 + 检出类型（中文名）+ 相似案例
- **批量诊断**：缩略图列表（点击放大）→ 批量诊断表格展示**预测类型（中文）与真实标签（英文+中文全名）对照**

## 6. 目录说明

```text
configs/protocols/ge20_ml.yaml    多标签协议（类型清单、划分参数）
configs/experiments/              训练配置
src/dram_diag/                    数据、协议、训练、推理、API 实现
scripts/build_multi_label_app.py  多标签标注站
scripts/convert_labels.py         标注 JSON → 多标签 CSV
scripts/audit_dataset.py          审计 + 生成 manifest
scripts/train.py                  训练入口
scripts/deploy_multilabel.py      生成部署清单（含检索索引）
scripts/serve.py                  FastAPI + Web 前端
artifacts/ge20_ml/                labels.csv、manifest、审计报告
artifacts/labeling/               标注站静态文件（data.json/index.html）
artifacts/deployment.json         部署清单
runs/                             训练产物（不提交）
legacy/                           历史项目归档，仅供参考
```

## 常见问题

- **批量诊断里图片"不存在"**：该图为未标注被排除的图（如 image_824/895），属正常
- **重新打标后如何更新**：重新导出 `label_v2.json` → 重跑第 3 步（指纹变化自动使旧 manifest 失效）→ 重训
- **调整阈值**：`deploy_multilabel.py --threshold 0.5`（检出判定线，可调）

详细设计见 `项目方案/PLAN.md`（历史方案文档，含评审记录）。
