# DRAM 晶圆缺陷诊断 v2

本项目按 `ge20-v2` 协议重建，主任务只分类原始样本数 `>=20` 的缺陷类。稳定性和准确性优先，原型约束分类器是主要算法增强，监督对比学习只作为可选实验。`legacy/` 仅作历史参考，不参与运行。

当前已冻结的数据结果：21 个分类类、523 张分类候选图；固定划分为训练 379、验证 48、校准已知 48、测试已知 48；另有校准未知 45、核心测试未知 50、压力测试未知 4，以及 78 类/528 张案例库。分类覆盖率应和 Macro-F1 一起报告。

> 本轮由 Codex 完成了代码、协议审计和自动化测试，但没有执行任何 smoke 或正式模型训练。以下训练和最终评估命令由项目负责人手动运行。

## 1. 环境准备

要求 Windows、PowerShell、Python 3.12。若现有 `.venv` 可用：

```powershell
Set-Location E:\Ai\dram-diag
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -r requirements.txt
python -m pip install -e . --no-build-isolation
python -m pytest -q
```

需要重新创建环境时：

```powershell
Set-Location E:\Ai\dram-diag
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -r requirements.txt
python -m pip install -e . --no-build-isolation
```

## 2. 数据审计与协议冻结

重新生成主协议清单：

```powershell
python scripts\audit_dataset.py --data-root 晶圆缺陷分类数据集 --protocol configs\protocols\ge20.yaml --out artifacts\ge20
```

核对 `artifacts/ge20/audit_report.json` 中的固定数字，并人工检查其中 7 组 `perceptual_hash_groups`。它们只是近重复候选，未经人工确认不得删除。正式训练前还需抽查：21 个分类类代表图、容易混淆的类别对，以及恰好 20 张的边界类。人工结论应另存为版本化审查记录；若改标签或删图，必须提升协议版本并重新生成 manifest，旧 checkpoint 随即失效。

## 3. Smoke 测试

只验证数据、损失、checkpoint 和加载链路，不用于比较模型：

```powershell
python scripts\train.py --config configs\experiments\baseline_imagenet.yaml --epochs 1 --seeds 42 --out-dir runs\smoke\baseline
python scripts\train.py --config configs\experiments\prototype_005.yaml --epochs 1 --seeds 42 --out-dir runs\smoke\prototype
```

## 4. 配置筛选：只用 seed 42

先独立比较归一化方式，不要同时改变其他因素：

```powershell
python scripts\train.py --config configs\experiments\baseline_imagenet.yaml --seeds 42
python scripts\train.py --config configs\experiments\baseline_gray.yaml --seeds 42
```

冻结较优归一化后，再筛选原型损失权重：

```powershell
python scripts\train.py --config configs\experiments\prototype_001.yaml --seeds 42
python scripts\train.py --config configs\experiments\prototype_005.yaml --seeds 42
python scripts\train.py --config configs\experiments\prototype_010.yaml --seeds 42
```

每轮最多保留两个候选。仅在稳定基线完成后才运行 SupCon：

```powershell
python scripts\train.py --config configs\experiments\supcon_optional.yaml --seeds 42
```

选模只看 validation Macro-F1，并同时检查 Balanced Accuracy、训练/验证差距和最差类别召回。不要读取 calibration 或 test 结果辅助选配置。

## 5. 入围配置：3 seeds

对每个入围配置运行固定划分的 42/43/44 三个 seed。以下用 `prototype_005` 举例：

```powershell
python scripts\train.py --config configs\experiments\prototype_005.yaml --seeds 42 43 44 --out-dir runs\finalists\prototype_005
```

根据三 seed 验证均值、标准差和过拟合情况锁定唯一配置，并在查看最终测试前决定使用单模型还是验证阶段已确定的集成。这里不是 5 折 × 3 seeds。

## 6. 最终配置稳定性复核：5 折 × 1 seed

只对唯一最终配置运行 development pool 的 5 折，固定 seed 42：

```powershell
python scripts\train.py --config configs\experiments\prototype_005.yaml --seeds 42 --folds 0 1 2 3 4 --out-dir runs\stability\prototype_005
```

5 折只覆盖 `train + validation`，不含校准集、测试集或未知类。若某折 Macro-F1 低于 5 折均值 0.10 以上，仅对该异常折补跑：

```powershell
python scripts\train.py --config configs\experiments\prototype_005.yaml --seeds 43 44 --folds 2 --out-dir runs\stability_reruns\prototype_005
```

正常流程总计是“固定划分 3 seeds + 5 folds × seed 42”；异常折才增加 seed 43/44。5 折结果只用于稳定性复核，不允许据此继续调超参数。

## 7. 锁定并执行一次最终评估

先根据验证结果确定 checkpoint 和单模型/集成形式。单模型示例：

```powershell
python scripts\lock_experiment.py --manifest artifacts\ge20\split_manifest.json --checkpoints runs\finalists\prototype_005\seed-42\best.pt --model-form single --alpha 0.5 --out artifacts\locks\ge20-final-v1.json
```

三模型集成示例（只有在查看测试结果前已决定集成时才可使用）：

```powershell
python scripts\lock_experiment.py --manifest artifacts\ge20\split_manifest.json --checkpoints runs\finalists\prototype_005\seed-42\best.pt runs\finalists\prototype_005\seed-43\best.pt runs\finalists\prototype_005\seed-44\best.pt --model-form ensemble --alpha 0.5 --out artifacts\locks\ge20-final-v1.json
```

确认锁文件后，最终评估只能执行一次：

```powershell
python scripts\evaluate_final.py --lock artifacts\locks\ge20-final-v1.json --manifest artifacts\ge20\split_manifest.json --out artifacts\reports\ge20-final-v1.json --bootstrap-iterations 2000
```

脚本用 calibration-known 拟合温度，用 calibration-known 与 calibration-unknown 选择拒识阈值；核心未知和压力未知不参与调参。成功后锁文件会标记 `final_test_consumed=true`，同一锁不能再次消费测试集，已有报告也不能覆盖。

## 8. 晋升与部署

闭集和开放集采用两道独立门槛：

```powershell
python scripts\promote.py --lock artifacts\locks\ge20-final-v1.json --report artifacts\reports\ge20-final-v1.json --manifest artifacts\ge20\split_manifest.json --out artifacts\deployment.json
```

可能得到三种模式：

- `classification_with_unknown_rejection`：闭集和开放集均通过，可启用未知拒识。
- `classification_review_only`：闭集通过、开放集未通过，只部署 21 类候选分类，低置信结果全部复核。
- `research_only`：闭集未通过，服务不会加载模型进行诊断。

启动服务：

```powershell
python scripts\serve.py --deployment artifacts\deployment.json --host 127.0.0.1 --port 8000
```

浏览器访问 `http://127.0.0.1:8000`。界面支持单图、批量、协议与覆盖信息、分类/原型证据、未知状态和相似案例。服务没有可用部署清单时只显示“模型未部署”，不会回退到旧模型。

## 9. `>=15` 覆盖率对照

只有主方案覆盖不足时才运行，不用于替代主协议的默认结论：

```powershell
python scripts\audit_dataset.py --protocol configs\protocols\ge15.yaml --out artifacts\ge15
python scripts\train.py --config configs\experiments\ge15_comparison.yaml --seeds 42
```

仅当 `>=15` 的闭集指标达到主方案门槛、与 `>=20` 的 Macro-F1/Balanced Accuracy 差距均不超过 0.05、最差类召回和未知误接收没有恶化，并确实增加业务必要类别时，才升级为候选。不同协议的 checkpoint、评估报告和部署清单禁止混用。

## 10. 目录说明

```text
configs/       数据协议和实验配置
src/dram_diag/ 新的数据、训练、评估、推理与 API 实现
scripts/       审计、训练、锁定、最终评估、晋升和启动命令
artifacts/     冻结 manifest、审计、实验锁、报告与部署清单
runs/          训练输出（不提交大 checkpoint）
tests/         协议、泄漏、模型、指标和检索测试
web/           新协议前端
legacy/        历史项目归档，仅供参考
```

详细设计和验收门槛见 `项目方案/PLAN.md`。
