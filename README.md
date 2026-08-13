# HPOD-Classifier

DRAM 晶圆缺陷 ROI 图像诊断原型。项目目标是对单张缺陷图像进行细分类、相似案例检索和开放集风险提示，并为后续接入经过专家审核的“缺陷族”映射保留接口。

## 当前状态

当前版本已经包含数据审计、固定开放集划分、图像预处理、ResNet18 模型封装、原型检索、开放集评分、推理接口、Gradio 界面骨架，以及完整的 ResNet18 训练循环（AMP 混合精度、AdamW、验证、最佳 checkpoint 保存）。训练命令可直接产出 `best.pt` 模型文件。

## 数据目录

默认数据目录如下：

```text
晶圆缺陷分类数据集/
├── label.csv
└── images/
    ├── image_0.jpg
    ├── image_1.jpg
    └── ...
```

`label.csv` 至少需要包含两列：

```text
IMAGE_NAME,DEFECT_ID
```

`IMAGE_NAME` 必须能在 `images/` 目录中找到对应文件。当前数据集实测为 1,150 张 `480×320` 单通道灰度 JPEG。

## 环境安装

建议使用 Python 3.12，并在项目根目录执行。

### 1. 创建虚拟环境

Windows PowerShell：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

如果 PowerShell 禁止执行激活脚本，可以只使用虚拟环境中的 Python：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 2. 安装 PyTorch

RTX 4070 Laptop 建议安装 CUDA 版本。若你的机器已经有可用的 `torch 2.6.0+cu124`，可跳过这一步。

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

验证 GPU：

```powershell
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

### 3. 安装项目依赖

```powershell
pip install -r requirements.txt
```

确认 Gradio：

```powershell
python -c "import gradio; print(gradio.__version__)"
```

### 4. 配置文件

默认参数由 `configs/default.yaml` 提供（`data_root`、`image_size`、`seed`、`batch_size`、`epochs`、`lr`、`num_workers`、`model`、`unknown_classes`、`tta`）。`train.py`、`evaluate.py` 会通过 `load_config()` 读取该文件；命令行参数可覆盖其中的默认值。

## 第一步：运行数据审计

在训练前先执行：

```powershell
python audit_dataset.py --data-root "晶圆缺陷分类数据集" --out artifacts --seed 42
```

审计脚本会检查：

- CSV 与图片是否一一对应
- 图片是否能正常解码
- 图像尺寸、通道和格式统计
- SHA-256 完全重复文件
- 每个 `DEFECT_ID` 的样本数量
- 高频、中频、长尾类别分档
- 校准未知类和测试未知类

输出文件：

```text
artifacts/
├── audit_report.json
├── defect_mapping.template.csv
└── split_manifest.json
```

其中 `split_manifest.json` 会记录可复现的数据划分。开放集协议为：

- 高频类别：样本数 `> 20`
- 中频类别：样本数 `5–20`
- 长尾类别：样本数 `< 5`
- 2 个类别作为校准未知类，用于确定拒识阈值
- 3 个独立类别作为测试未知类，只用于最终测试

如果某个类别档位不足，脚本会报错，不会静默改变实验协议。

## 第二步：预处理规则

预处理规则：

- 原始灰度图复制成 3 通道
- 使用 `320×224` 输入尺寸
- 使用等比例 letterbox，禁止直接拉伸
- 保留原图 3:2 比例，避免 224×224 带来的大面积无效边框
- 按 ImageNet 均值/方差做标准化（`mean=[0.485,0.456,0.406]`，`std=[0.229,0.224,0.225]`）

这些规则由 `src/dram_diag/data.py` 的 `image_array()` 统一实现，训练与推理共用，无需单独验证。

## 第三步：训练

标准训练命令：

```powershell
python train.py --data-root "晶圆缺陷分类数据集" --epochs 20 --batch-size 32 --device cuda
```

`train.py` 会先运行数据审计并生成划分，再开始 epoch 训练。可用参数（默认值来自 `configs/default.yaml`）：

| 参数 | 默认 |
|---|---|
| `--data-root` | `晶圆缺陷分类数据集` |
| `--epochs` | `20` |
| `--batch-size` | `32` |
| `--lr` | `3e-4` |
| `--device` | `cuda`（GPU 不可用时自动回退 CPU） |
| `--out-dir` | `runs/hpod-resnet18` |
| `--seed` | `42` |

首次训练会自动下载 ResNet18 预训练权重（约 45 MB），缓存到 `~/.cache/torch`，之后无需重复下载。

预期训练产物：

```text
runs/hpod-resnet18/
├── best.pt        # 最佳验证准确率 checkpoint（state_dict + class_to_idx + config）
└── history.json   # 每个 epoch 的 loss / train_acc / val_acc
```

8GB 显存建议从以下配置开始：

- 输入：`320×224`
- batch size：`16` 或 `32`
- ResNet18
- AMP 混合精度开启
- 首轮训练 20 个 epoch

如果显存不足，将 batch size 调到 `8` 或 `16`，不要修改数据划分。

## 第四步：评估

当前 `evaluate.py` 可以重新生成并打印审计/划分信息：

```powershell
python evaluate.py --data-root "晶圆缺陷分类数据集"
```

完整评估模块完成后，将使用模型 checkpoint 对已知测试类和测试未知类分别报告：

- Accuracy、Macro-F1、Balanced Accuracy、Top-5 Accuracy
- AUROC、AUPR、FPR@95TPR、OSCR
- Recall@5、mAP@5、NDCG@5
- ECE 与可靠性曲线
- 单次前向延迟、TTA 延迟、吞吐量和显存占用

检索指标中的 ground truth 是查询图像的 `DEFECT_ID`。Top-5 中出现同 `DEFECT_ID` 案例才算命中；测试未知类没有同类案例时，单独报告无案例覆盖情况。

## 第五步：启动 Gradio 界面

```powershell
python app.py
```

启动后终端会显示本地访问地址，通常是：

```text
http://127.0.0.1:7860
```

当前界面包含：

- 单图上传和诊断结果 JSON
- 批量导入页面的基础占位

在接入训练好的 checkpoint 之前，界面会显示未训练状态，不能作为实际分类结果使用。后续接入模型后，界面将加载模型、原型库和校准阈值，显示类别、置信度、未知分数、Top-5 案例和人工复核标志。

## 结果导出

如果已有 JSON 结果文件，可以转换为 CSV：

```powershell
python export_results.py artifacts/results.json --output artifacts/results.csv
```

## 缺陷族映射审核

打开：

```text
artifacts/defect_mapping.template.csv
```

由专业人员补充 `defect_name` 和 `defect_family`，并将 `mapping_status` 改为 `approved`。在映射审核完成前，系统不会自动推断工艺语义，也不会启用族分类损失或族一致性证据。

## 常见问题

### 找不到 `torch`

确认虚拟环境已激活，然后重新安装 PyTorch：

```powershell
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

### 找不到 `gradio`

```powershell
python -m pip install gradio
```

### CUDA 显示 False

这通常表示安装了 CPU 版 PyTorch，或显卡驱动/CUDA wheel 不匹配。先执行：

```powershell
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

### 审计提示类别不足

当前协议要求至少存在 2 个高频、2 个中频和 1 个长尾类别。如果你更换了数据集，需要重新确认类别分档或手工指定开放集类别，不能直接混用旧的 `split_manifest.json`。

## 当前限制

- 当前评估模块尚未实现完整的已知/未知类指标与检索指标报告
- 当前 Gradio 页面尚未接入训练模型和完整批量复核流程
- 当前没有边界框、分割掩码、wafer/lot 元数据或工艺根因诊断
- 缺陷族必须经过专家审核，不能由模型自动编造
