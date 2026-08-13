# HPOD-Classifier

DRAM 晶圆缺陷 ROI 图像的层级原型开放集分类原型。

## 快速开始

```powershell
pip install -r requirements.txt
python audit_dataset.py --data-root 晶圆缺陷分类数据集
python train.py --dry-run
python app.py
```

审计会生成 `artifacts/audit_report.json`、`artifacts/split_manifest.json` 和待审核的缺陷族映射模板。开放集协议将 2 个类别用于校准未知阈值，3 个独立类别仅用于最终测试。

默认输入为 320×224，保持原始 3:2 比例；TTA 默认关闭，单次前向与 TTA 性能分别统计。
