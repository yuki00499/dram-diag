# DRAM 缺陷分类项目重构方案：>=20 高精度分类协议

> 状态：方案草稿，已完成两轮一致性审查，待人工完善  
> 主协议：`ge20-v2`  
> 对照协议：`ge15-v2`

## 1. 目标与结论

当前根目录仅保留数据集和 `legacy/` 历史归档。新项目不继续修补旧代码，而是建立干净、可复现、职责明确的新工程。

主方案采用“高精度小范围分类”：

- 开放集改为分层协议：校准未知 `[7, 2]`，核心测试未知 `[6, 65, 1]`，长尾压力测试未知 `[51]`。
- 排除未知类后，仅将样本数 `>=20` 的类别纳入分类头。
- 当前数据得到 **21 个分类类**、523 张分类候选图、379 张训练图。
- 其余已知类别进入案例库和人工复核，不参与分类损失。
- `>=15` 作为覆盖率对照协议，不作为首选生产方案。

当前数据统计：

| 协议 | 分类类 | 原始样本 | 训练样本 | 每个已知留出集总量 | 平均每类/集 | 已知样本覆盖率 |
|---|---:|---:|---:|---:|---:|---:|
| `>=15` | 28 | 646 | 460 | 62 | 2.21 | 61.5% |
| **`>=20`** | **21** | **523** | **379** | **48** | **2.29** | **49.8%** |

`>=20` 的最小类别有20张，按当前划分约有14张进入训练；`>=15` 的15张类别只有约9张训练样本。因此当前模型约50% Macro-F1 的情况下，`>=20` 更适合作为高精度主方案。

## 2. 新工程结构

历史目录 `legacy/` 只读保留，不作为新代码依赖。

```text
dram-diag/
├─ data/
│  └─ raw/                         # 配置直接指向原位数据集，或复制数据；不要求 Windows 软链接
├─ configs/
│  ├─ protocols/ge20.yaml          # 主协议
│  └─ protocols/ge15.yaml          # 对照协议
├─ src/dram_diag/
│  ├─ data_audit.py                # 数据审计与数据版本
│  ├─ protocol.py                  # 划分、类别角色、manifest
│  ├─ dataset.py                   # 预处理与 Dataset
│  ├─ model.py                     # 编码器、分类头、嵌入头
│  ├─ train.py                     # 训练入口
│  ├─ evaluate.py                  # 闭集、开放集、检索评估
│  ├─ rejection.py                 # 拒识校准
│  ├─ retrieval.py                 # 案例库与检索
│  └─ schemas.py                   # checkpoint、预测、报告结构
├─ scripts/
│  ├─ audit_dataset.py
│  ├─ train.py
│  ├─ evaluate.py
│  └─ build_case_index.py
├─ tests/
│  ├─ test_protocol.py
│  ├─ test_dataset.py
│  ├─ test_evaluation.py
│  └─ test_retrieval.py
├─ artifacts/
│  ├─ ge20/
│  └─ ge15/
├─ runs/
│  ├─ ge20/
│  └─ ge15/
└─ docs/
```

第一阶段不迁移旧 Web 页面和旧训练脚本。模型协议稳定后，再重新设计前端接口。

## 3. 数据协议

### 3.1 类别角色

manifest 必须记录以下互斥角色：

- `classification_classes`：达到门槛且不属于未知类。
- `case_library_classes`：已知但未达到门槛，仅用于相似案例和人工复核。
- `calibration_unknown_classes`：`[7, 2]`，共45张，只用于拒识阈值校准。
- `test_unknown_core_classes`：`[6, 65, 1]`，分别13、8、29张，共50张，用于核心开放集指标。
- `test_unknown_stress_classes`：`[51]`，4张，只用于长尾未知压力测试，不参与硬门槛。

所有图片必须只出现在一个 split 中，所有类别必须恰好属于一个角色。

### 3.2 划分方式

主协议使用：

- `train`：约70%
- `validation`：约10%
- `calibration_known`：约10%
- `test_known`：约10%
- 未知类全量进入 `calibration_unknown`、`test_unknown_core` 或 `test_unknown_stress`
- 案例库类别全量进入 `case_library`

其中 `train + validation` 组成只用于最终稳定性复核的 `development_pool`；`calibration_known` 专用于概率/拒识校准，`test_known` 始终锁定。5折索引从 development pool 派生，不改变原始 split 归属。

每个 manifest 保存：

```json
{
  "protocol_version": "ge20-v2",
  "min_class_count": 20,
  "split_seed": 42,
  "dataset_fingerprint": "...",
  "manifest_fingerprint": "...",
  "classification_classes": [],
  "case_library_classes": [],
  "calibration_unknown_classes": [],
  "test_unknown_core_classes": [6, 65, 1],
  "test_unknown_stress_classes": [51],
  "splits": {},
  "cv_folds": []
}
```

数据审计必须检查 CSV/图片一致性、图片可解码性、重复图片、类别计数、划分泄漏和数据指纹，并在 manifest 和报告中逐类记录未知类样本数。未知类清单变化必须提升协议版本号，旧 checkpoint 不得跨协议评估或部署。

## 4. 模型方案：稳定基线与算法亮点

算法亮点必须建立在可复现的普通分类基线上，不能通过一次性叠加复杂损失换取不可复现的指标。增强模块先做单 seed 筛选，只有入围配置才进行三 seed 验证和失败回退检查。

### 4.1 稳定基线（Baseline）

- Backbone：预训练 ResNet18；另保留 ConvNeXt-Tiny 作为单独架构对照，不在首轮混用。
- 输入：等比例 letterbox，480x320，灰度复制为三通道。
- 归一化在基线阶段做单因素对照：ImageNet mean/std 与仅由训练集计算的 `dataset_gray` mean/std；后者统计值随 manifest 保存，最终默认值由三 seed 验证结果决定。
- 输出：21维分类 logits 和归一化 embedding。
- 损失：Cross Entropy，配合轻度类别均衡采样。
- 增强：轻度旋转、亮度和对比度调整，不使用可能破坏小缺陷主体的强擦除。
- 使用 validation Macro-F1 保存最佳 checkpoint，训练 seed 固定为 `42/43/44`。
- 记录单 seed、三 seed 均值、标准差和最差类别，不以单次最高分作为结论。

### 4.2 低风险算法亮点 A：原型约束分类器

在共享 embedding 上为每个分类类维护归一化类别原型，分类头仍保留，避免完全依赖最近邻：

```text
logits = classification_head(embedding)
prototype_score = cosine(embedding, class_prototypes)
loss = CE(logits, y) + lambda_proto * prototype_loss
```

原型只由训练集计算，不能使用验证、校准或测试图像。`prototype_loss` 采用类内拉近、类间间隔约束；`lambda_proto` 固定消融网格为 `0.01/0.05/0.1`，仅在 validation Macro-F1、Balanced Accuracy 和 seed 稳定性都不下降时启用。

该模块同时服务于分类和开放集拒识，是本项目的主要算法亮点：分类头负责判别，原型距离负责验证“输入是否真的接近该类别”。推理时保留两个证据：分类概率和最近原型相似度。

### 4.3 低风险算法亮点 B：类别感知监督对比学习

仅在基线和原型约束稳定后加入 supervised contrastive loss，使用类别均衡 batch，避免少数类别在对比学习中没有正样本。总损失为：

```text
L = L_ce + lambda_proto * L_proto + lambda_supcon * L_supcon
```

`lambda_supcon` 从 `0.05` 开始做单因素消融。若训练集准确率更快达到100%而验证指标下降，则自动淘汰该配置。

### 4.4 稳定性增强与模型选择

- 第一轮配置筛选统一使用 seed 42 和固定 train/validation，只比较 validation 指标；每轮最多保留两个候选进入三 seed 阶段。
- 入围配置使用 seed `42/43/44`，按 validation Macro-F1 均值优先、Balanced Accuracy 次优、标准差诊断的顺序选出唯一最终配置。
- 首选单模型。三 seed 概率集成只有在 validation 上比预先选定的最佳单模型 Macro-F1 提升至少0.02、Balanced Accuracy 不下降且推理成本可接受时才被锁定。
- 单模型或集成形态必须在查看 `test_known` 前锁定；禁止同时测试多种形态后选择测试分数最高者。
- Dropout、label smoothing、冻结 backbone 和类别加权作为正则化消融，不与 SupCon 同时首次启用。
- 默认前5个 epoch 冻结 backbone，只训练分类头；随后解冻并使用 backbone `1e-4`、分类头 `3e-4` 的分组学习率和 cosine 衰减。冻结策略与全量微调作为单因素对照。
- 所有模型必须保存配置、类别映射、协议指纹、训练 seed、原型版本和损失权重。
- 训练过程增加 train/validation gap 监控；若训练准确率连续达到100%且验证 Macro-F1 无改善，提前停止并标记过拟合。

### 4.5 暂缓的算法方向

缺陷族层次化分类、自动聚类和在线增量学习暂不进入第一版。当前没有经过专家审核的缺陷族映射，贸然加入层次损失会把标签噪声引入主任务。待21类基线和原型模型稳定后，再作为独立研究扩展。

## 5. 评估与验收

### 5.1 闭集指标

只对21个分类类计算：

- Accuracy、Macro-F1、Balanced Accuracy、Top-5 Accuracy；
- 每类 Precision、Recall、F1；
- 混淆矩阵；
- ECE 和可靠性曲线。

由于每类测试样本只有约2至3张，最终报告必须给出三 seed 均值和标准差、每类支持度、按类别分层的 bootstrap 95% CI 以及最差5个类别召回率。bootstrap 固定2000次并记录随机种子。

三 seed 标准差作为稳定性诊断，不再单独作为硬门槛。最终候选还需执行分层5折交叉验证作为稳定性复核，规则如下：

- 5折仅使用 `development_pool = train + validation`，不得包含 `calibration_known`、`test_known` 或任何未知类。
- 最终配置固定后，使用 seed 42 运行5折，每折约80% development 数据训练、20%验证。
- 这是 **5折 x 1 seed**，不是5折 x 3 seeds；加上此前固定划分的三 seed，最终配置通常共8次正式训练。
- 5折结果不得继续用于选择损失权重、归一化、backbone 或其他超参数，只判断固定配置是否依赖单次划分。
- 若某一折 Macro-F1 低于5折均值0.10以上，只对该异常折补跑 seed 43/44；不默认扩展为15次训练。
- 交叉验证折索引、seed 和数据指纹写入 manifest 派生产物，保证复现且不改变锁定测试集。
- 若最终配置未通过5折门槛，本实验版本停止，不得根据5折结果回头挑选其他配置；后续改进必须建立新实验版本。

### 5.2 开放集指标

拒识分数初版只融合最大分类概率和最近分类原型距离。阈值只能使用 `calibration_known` 与 `[7,2]` 校准未知类选择；核心测试未知和长尾压力未知不得参与调参。

核心测试集 `[6,65,1]` 报告样本加权和类别宏平均 AUROC、AUPR、OSCR、unknown recall、unknown false accept rate、known false reject rate 及 bootstrap 95% CI。FPR@95TPR 继续报告但不设硬门槛。51类单独报告逐样本分数、unknown recall 和宽置信区间，不与核心指标混合。

### 5.3 案例检索

- 分类原型只使用21类训练样本。
- 案例索引使用训练样本和降级类别案例。
- 开放集拒识不得使用降级案例原型。
- 查询自身不得作为相似案例返回。

报告 Recall@5、mAP@5、NDCG@5 和降级类人工复核覆盖率。

### 5.4 主方案验收门槛

验收拆成两道门槛。先判断21类闭集分类是否可部署：

- Macro-F1 `>=0.70`；
- Macro-F1 分层 bootstrap 95% CI 下限 `>=0.60`；
- Balanced Accuracy `>=0.70`；
- Top-5 Accuracy `>=0.85`；
- ECE `<=0.10`；
- 5折稳定性复核 Macro-F1 均值 `>=0.65`，最低单折 `>=0.50`；
- 每类 Recall 只作为诊断项，当前测试支持度不足以设置可信的逐类硬门槛。

只有闭集门槛通过后，才判断自动未知拒识是否可启用：

- 核心未知 AUROC `>=0.80`，bootstrap 95% CI 下限 `>=0.70`；
- 核心未知类别宏平均 unknown recall `>=0.70`；
- 校准阈值下 known false reject rate `<=0.20`，核心未知 false accept rate `<=0.30`；
- FPR@95TPR 与51类压力结果只报告，不作为硬门槛。

若闭集通过而开放集未通过，可以部署“21类候选分类 + 全部低置信输入人工复核”，但不得宣称具备可靠未知检测。两道门槛均未通过时，模型只能作为研究结果。

### 5.5 测试集锁定规则

- `test_known`、`test_unknown_core` 和 `test_unknown_stress` 在 manifest 生成后锁定。
- 模型配置、checkpoint 选择规则、单模型/集成形态、拒识公式和校准流程必须在读取测试结果前确定。
- 测试前生成 `experiment_lock.json`，写入上述冻结决策、候选 checkpoint 哈希、代码版本和预期验收规则；评估脚本只接受该清单指定的模型。
- 每个正式协议只生成一次最终测试报告；失败后不得继续根据测试结果调参并重复覆盖报告。
- 若最终测试失败，当前 test 已被消费。下一轮即使提升实验版本，也只能把同一 test 分数作为“已见测试集比较”，不得再宣称是无偏最终结果；重新进行部署验收需要新增数据或重新保留从未查看的测试类别/样本。
- 测试脚本保存运行时间、代码版本、协议指纹和输出哈希，模型晋升工具只接受锁定报告。

### 5.6 `>=15` 对照协议

只有在以下条件同时满足时才考虑使用 `>=15`：

- `>=20` 覆盖率明显不足；
- `>=15` 的 Macro-F1、Balanced Accuracy 与 `>=20` 差距不超过0.05；
- `>=15` 的最差类别 Recall 和开放集误接收率不恶化；
- `>=15` 能覆盖业务中必须保留的类别。

`>=15` 明确定位为覆盖率对照，不要求达到与 `>=20` 相同的绝对门槛，也不得因覆盖率更高直接替代主方案。只有其闭集指标达到主方案门槛且相对 `>=20` 下降不超过0.05时，才能进入部署候选。

## 6. 前端与预测接口

### 6.1 前端策略

沿用 `legacy` 版本的单图诊断、批量选择、结果表格、缩略图预览和 CSV 导出交互，不直接复制旧后端耦合代码。新前端通过稳定的预测 JSON 接口接入新模型，保留原有用户操作习惯。

在旧界面基础上增加：

- “可信分类 / 待复核 / 未知”状态色标；
- 分类概率与最近原型相似度两个证据条；
- Top-5 分类候选与 Top-5 相似案例分栏展示；
- 当前协议、模型版本、分类覆盖范围和复核数量；
- 批量结果中的 `unknown_score`、`review_required` 和相似案例入口。

前端不展示未经专家确认的工艺根因，不把模型置信度包装成确定性结论。若新接口暂时不可用，前端只显示模型未就绪，不回退到旧模型静默推理。

### 6.2 预测接口

统一预测结果结构：

```json
{
  "image_name": "image_001.jpg",
  "predicted_class": 23,
  "confidence": 0.87,
  "unknown_score": 0.12,
  "status": "known_confident",
  "review_required": false,
  "top5_candidates": [],
  "top5_similar_cases": [],
  "model_version": "ge20-resnet18-v2"
}
```

状态只允许：`known_confident`、`known_uncertain`、`unknown`、`low_quality`；是否进入复核队列单独由布尔字段 `review_required` 表示。

对案例库类别不得强制输出21类中的错误细分类；低证据输入必须进入人工复核并附带相似案例。

## 7. 实施顺序

1. 归档 `legacy/`，建立全新目录结构。
2. 实现数据审计和 `ge20-v2` manifest，并核对核心未知/压力未知样本数量。
3. 人工抽查21个分类类的代表图、类间相似对和20张边界类别，记录视觉可分性与疑似标签噪声。
4. 实现21类 ResNet18 稳定基线和单元测试。
5. 运行1 epoch smoke test，确认分类、原型索引和 checkpoint 可加载。
6. 独立对比 ImageNet 与 `dataset_gray` 归一化，冻结选择后不再随其他模块变更。
7. 使用 seed 42 筛选基线、正则化、原型约束和监督对比学习配置，每轮最多保留两个候选。
8. 对入围配置运行固定划分的三 seed，锁定唯一最终配置和单模型/集成形态。
9. 对最终配置运行 development pool 的5折 x 1 seed 稳定性复核；只在异常折补跑两个 seed。
10. 完成开放集校准，锁定拒识公式和阈值选择流程。
11. 只运行一次锁定测试集评估，并生成分层 bootstrap、闭集、开放集与检索报告。
12. 只有主方案未达到覆盖目标时，才运行 `>=15` 覆盖率对照实验。
13. 将旧前端迁移到新预测 API，增加证据面板、拒识状态和相似案例展示。
14. 最后再加入人工复核、批量导出和模型版本管理功能。

### 7.1 实验训练次数预算

| 阶段 | 训练方式 | 用途 |
|---|---:|---|
| Smoke | 1 epoch x 1 seed | 验证代码和产物链路 |
| 配置筛选 | 每配置 x seed 42 | 低成本淘汰无效配置 |
| 候选确认 | 每个入围配置 x 3 seeds | 比较均值与随机初始化波动 |
| 最终稳定性 | 最终配置 x 5 folds x seed 42 | 检查固定划分依赖 |
| 异常折复核 | 仅异常折 x seed 43/44 | 判断异常来自数据还是初始化 |

默认不执行5折 x 3 seeds。只有多个折均出现无法解释的高波动，且项目需要论文级重复性分析时，才另立实验版本扩展到15次训练。

## 8. 默认假设

- `legacy/` 中所有代码和文档只作为参考，不直接复制为新项目架构。
- 开放集采用核心未知与长尾压力测试分离的 `ge20-v2`；历史固定清单保留在报告中，但不以4张的51类决定硬门槛。
- 若核心未知类样本不足或数据审计发现异常，允许更换未知类别，但必须提升协议版本并废弃旧协议 checkpoint 的直接比较。
- 主生产协议为 `>=20`，`>=15` 为对照协议。
- 目标是高精度小范围分类，不追求覆盖全部缺陷类别。
- 稳定性和准确性优先于算法复杂度；复杂模块必须证明收益后才能保留。
- 原型约束分类器是首选算法亮点，监督对比学习是后续可选增强。
- 前端复用旧版交互设计，但后端接口、模型状态和证据展示使用新协议。
- 缺陷名称和缺陷族必须由人工审核，系统不自动推断工艺语义。
- 在模型达到绝对验收门槛前，不启动自动部署和自动改判。

## 9. 测试与产物验收

### 9.1 数据与协议测试

- `ge20-v2` 必须得到21个分类类、379张训练、validation/calibration-known/test-known 各48张。
- calibration-unknown 45张、core test-unknown 50张、stress test-unknown 4张、case library 78类共528张。
- 验证类别角色互斥、图片全量覆盖、无重复文件跨 split、manifest 指纹防篡改。
- `dataset_gray` 统计只能读取训练 split；CV 模式只能读取对应折的训练部分。

### 9.2 训练与评估测试

- checkpoint 必须保存类别映射、协议指纹、归一化统计、seed、fold、损失权重和模型版本。
- 协议或类别映射不一致时，评估、集成和模型晋升必须失败。
- bootstrap 固定2000次并可复现；5折不得读取 calibration/test 数据。
- 测试报告已存在时默认拒绝覆盖，必须显式创建新实验版本。
- 原型仅由训练数据构建；案例检索排除查询自身；降级案例不得进入开放集分类原型。

### 9.3 API 与前端测试

- API 输出真实 `DEFECT_ID`，不输出分类头内部索引。
- 分别验证四种状态、`review_required`、Top-5 分类候选和相似案例。
- 未通过开放集门槛时，前端不得显示“未知检测已启用”。
- 批量导出保留协议版本、模型版本、未知分数和复核状态。

## 10. 评审意见处理记录

| 评审项 | 处理结论 | 方案调整 |
|---|---|---|
| 硬伤1：51类只有4张 | 采纳，且提升协议版本 | 核心未知改为 `[6,65,1]` 共50张；51类改为报告型长尾压力测试；分类与未知拒识分开验收 |
| 硬伤2：已知测试集过小 | 采纳 | 三 seed std 改为诊断项；加入分层 bootstrap；最终配置使用 development pool 做5折 x 1 seed复核，不默认执行15次训练 |
| 硬伤3：归一化未经验证 | 采纳 | 基线阶段独立比较 ImageNet 与 `dataset_gray`，训练集统计写入 manifest/checkpoint |
| Windows 软链接风险 | 采纳 | 配置直接指向原位数据集，软链接不再是前提 |
| 原型损失权重不明确 | 采纳 | 固定消融网格 `0.01/0.05/0.1` |
| 冻结策略不明确 | 采纳 | 默认冻结 backbone 5轮，再使用分组学习率解冻微调 |
| 视觉可分性复核缺失 | 采纳 | 正式训练前人工抽查21类、相似类别对和20张边界类 |
| 5折与3 seed关系不明确 | 采纳 | 明确为固定划分3 seed加最终配置5折 x 1 seed，默认8次；仅异常折补 seed |
| 测试集可能参与模型/集成选择 | 采纳 | 新增 `experiment_lock.json`，单模型/集成形态先冻结，测试只运行一次 |

说明：原审查中“未知侧分辨率仅1/4”的说法不完全准确，因为旧 test-unknown 合计46张；真正风险是未知类别种类仅3个，且51类只有4张，无法稳定评估跨未知类别泛化。本方案按类别层面风险处理，而不是只增加样本加权指标。
