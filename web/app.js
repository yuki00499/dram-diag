const $ = (selector) => document.querySelector(selector);
const state = {
  model: null,
  diag: { split: "", offset: 0, total: 0, selected: null },
  batch: { mode: "dataset", split: "", files: [], results: null, source: "", previewUrls: {} },
};
const escapeHtml = (value) => String(value ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const SPLIT_ZH = { development: "开发集", train: "训练集", validation: "验证集", test_known: "测试集" };

const STATUS_META = {
  known_confident: ["可信检出", "green", "可直接采用"],
  known_uncertain: ["置信偏低", "gold", "需人工复核"],
  no_detection: ["无检出", "gray", "需人工复核"],
  unknown: ["未知模式", "red", "需人工复核"],
  low_quality: ["图像质量不足", "orange", "需人工复核"],
};

const PAGE_META = {
  overview: ["数据概览", "协议信息、缺陷族分布与数据划分"],
  evaluation: ["模型评估", "OOF 证据、标签错误与稳定性分析"],
  diagnose: ["单图诊断", "分类证据、真实标签对照与相似案例"],
  batch: ["批量诊断", "数据集抽样或上传批量，筛选、对照与导出"],
};

async function api(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) {
    let message = response.statusText;
    try { message = (await response.json()).detail || message; } catch (_) {}
    throw new Error(message);
  }
  return response.json();
}

function switchView(name) {
  document.querySelectorAll(".view").forEach(item => item.classList.add("hidden"));
  $("#view-" + name).classList.remove("hidden");
  document.querySelectorAll(".nav-btn").forEach(item => item.classList.toggle("active", item.dataset.view === name));
  $("#view-title").textContent = PAGE_META[name][0];
  $("#view-desc").textContent = PAGE_META[name][1];
  if (name === "diagnose" && !state.diag.loaded) loadDiagGrid(true);
}

function badgeHtml(status) {
  const meta = STATUS_META[status];
  if (!meta) return `<span class="status-badge gray">${escapeHtml(status)}</span>`;
  return `<span class="status-badge ${meta[1]}">${meta[0]}</span>`;
}

function metricHtml(items) {
  return items.map(([k, v]) => `<div class="metric-item"><div class="k">${escapeHtml(k)}</div><div class="v">${v}</div></div>`).join("");
}

function configureSplitOptions(splitSizes) {
  const available = Object.entries(splitSizes || {}).filter(([, count]) => Number(count) > 0);
  const optionHtml = available.map(([value, count]) => `<option value="${escapeHtml(value)}">${escapeHtml(SPLIT_ZH[value] || value)}（${count}）</option>`).join("");
  ["#diag-split", "#batch-split"].forEach(selector => {
    const select = $(selector);
    const previous = select.value;
    select.innerHTML = optionHtml || '<option value="">暂无数据</option>';
    select.value = available.some(([value]) => value === previous) ? previous : (available[0]?.[0] || "");
  });
  state.diag.split = $("#diag-split").value;
  state.batch.split = $("#batch-split").value;
}

/* ---------- 数据概览 ---------- */
async function loadOverview() {
  try {
    const model = state.model = await api("/api/model/info");
    const ready = model.ready;
    $("#status-dot").className = "status-dot " + (ready ? "ok" : "warn");
    $("#status-text").textContent = ready ? "就绪" : "模型未部署";
    $("#side-model-status").textContent = ready
      ? "● " + (model.model_version || "已部署")
      : "● 未部署（无可用模型）";
    $("#side-model-status").className = "model-status " + (ready ? "ok" : "warn");

    const inputSize = (model.input_size || [480, 320]).join(" × ");
    $("#ov-metrics").innerHTML = metricHtml([
      ["协议版本", escapeHtml(model.protocol_version)],
      ["缺陷族数", `${model.num_classes} <em>类</em>`],
      ["标注样本", `${model.classification_samples} <em>/ ${model.total_samples}</em>`],
      ["图像尺寸", inputSize],
      ["设备", escapeHtml(model.device)],
    ]);

    const dist = await api("/api/model/class-distribution");
    configureSplitOptions(dist.split_sizes);
    const entries = Object.entries(dist.type_counts || {}).sort((a, b) => Number(a[0]) - Number(b[0]));
    $("#ov-total").textContent = `${entries.reduce((s, [, c]) => s + Number(c), 0)} 正样本`;
    const names = model.type_names_full || model.type_names || {};
    const values = entries.map(([id, count]) => ({ label: names[id] || ("类型 " + id), value: Number(count) }));
    if (values.length) horizontalBarChart($("#chart-dist"), values);

    const splitRows = Object.entries(dist.split_sizes || {}).map(([name, count]) => {
      const zh = SPLIT_ZH[name] || name;
      return `<tr><td>${zh}</td><td class="mono">${count}</td></tr>`;
    }).join("");
    $("#ov-splits").innerHTML = `<thead><tr><th>集合</th><th>样本数</th></tr></thead><tbody>${splitRows}</tbody>`;

    const typeRows = entries.map(([id, count]) => `<tr><td class="mono">${escapeHtml(id)}</td><td>${escapeHtml(names[id] || "—")}</td><td class="mono">${count}</td></tr>`).join("");
    $("#ov-types").innerHTML = `<thead><tr><th>类型</th><th>名称（中英对照）</th><th>正样本数</th></tr></thead><tbody>${typeRows}</tbody>`;
  } catch (error) {
    $("#status-text").textContent = "服务不可用";
    $("#status-dot").className = "status-dot";
  }
}

/* ---------- 模型评估 ---------- */
function evalNumber(value, digits = 4) {
  return value == null || !Number.isFinite(Number(value)) ? "—" : Number(value).toFixed(digits);
}

function evalEmpty(selector, text = "暂无数据") {
  const element = $(selector);
  if (element) element.innerHTML = `<div class="empty">${escapeHtml(text)}</div>`;
}

async function loadEvaluation() {
  try {
    const model = state.model || (state.model = await api("/api/model/info"));
    const evaluation = await api("/api/model/evaluation");
    const names = model.type_names_full || model.type_names || {};
    const coreLabels = [2, 3, 4, 5];

    if (evaluation) {
      const ci = evaluation.core_macro_f1_ci;
      $("#ev-metrics").innerHTML = metricHtml([
        ["核心 Macro-F1", evalNumber(evaluation.core_macro_f1 ?? evaluation.macro_f1)],
        ["全标签 Macro-F1", evalNumber(evaluation.macro_f1)],
        ["Micro-F1", evalNumber(evaluation.micro_f1)],
        ["Sample-F1", evalNumber(evaluation.sample_f1)],
        ["mAP", evalNumber(evaluation.map)],
        ["Exact Match", evalNumber(evaluation.exact_match)],
        ["Hamming Loss", evalNumber(evaluation.hamming_loss)],
        ["核心 Macro-F1 CI", ci ? `${evalNumber(ci.lower)}–${evalNumber(ci.upper)}` : "—"],
      ]);

      const ids = Object.keys(evaluation.per_label_f1 || {}).sort((a, b) => Number(a) - Number(b));
      const perLabel = ids.map(id => {
        const confusion = (evaluation.confusion || {})[id] || {};
        return `<tr><td class="mono">${escapeHtml(id)}</td><td>${escapeHtml(names[id] || ("类型 " + id))}</td>` +
          `<td class="mono">${confusion.support ?? evaluation.support?.[id] ?? "—"}</td>` +
          `<td class="mono">${evalNumber(confusion.precision ?? evaluation.per_label_precision?.[id])}</td>` +
          `<td class="mono">${evalNumber(confusion.recall ?? evaluation.per_label_recall?.[id])}</td>` +
          `<td class="mono">${evalNumber(evaluation.per_label_f1?.[id])}</td>` +
          `<td class="mono">${evalNumber(evaluation.per_label_pr_auc?.[id])}</td>` +
          `<td class="mono">${evalNumber(evaluation.per_label_auc?.[id])}</td>` +
          `<td class="mono">${evalNumber(evaluation.thresholds?.[id], 2)}</td></tr>`;
      }).join("");
      $("#ev-perlabel").innerHTML = `<thead><tr><th>类型</th><th>类别</th><th>支持数</th><th>Precision</th><th>Recall</th><th>F1</th><th>PR-AUC</th><th>ROC-AUC</th><th>阈值</th></tr></thead><tbody>${perLabel}</tbody>`;
      $("#ev-split-meta").textContent = `${evaluation.source === "oof" ? "5折 OOF" : "验证集"} · ${evaluation.fold_core_macro_f1_std == null ? "当前部署评估" : `std ${evalNumber(evaluation.fold_core_macro_f1_std)}`}`;

      const confusionRows = ids.map(label => ({ label, ...((evaluation.confusion || {})[label] || {}) }));
      if (confusionRows.length) {
        confusionHeatmap($("#chart-confusion"), confusionRows, names);
        $("#ev-confusion").innerHTML = `<thead><tr><th>类型</th><th>TN</th><th>FP</th><th>FN</th><th>TP</th><th>Precision</th><th>Recall</th><th>F1</th></tr></thead><tbody>` +
          confusionRows.map(row => `<tr><td>${escapeHtml(names[row.label] || `类型 ${row.label}`)}</td><td>${row.tn ?? "—"}</td><td class="metric-fp">${row.fp ?? "—"}</td><td class="metric-fn">${row.fn ?? "—"}</td><td class="metric-tp">${row.tp ?? "—"}</td><td>${evalNumber(row.precision)}</td><td>${evalNumber(row.recall)}</td><td>${evalNumber(row.f1)}</td></tr>`).join("") + `</tbody>`;
      } else { evalEmpty("#ev-confusion"); }

      const curves = evaluation.pr_curves || {};
      const validCurves = Object.fromEntries(Object.entries(curves).filter(([, points]) => points && points.length));
      if (Object.keys(validCurves).length) {
        prChart($("#chart-pr"), validCurves, names, coreLabels);
        $("#ev-pr-meta").textContent = `${Object.keys(validCurves).length} 个标签 · 核心标签优先`;
      } else {
        $("#chart-pr").setAttribute("viewBox", "0 0 640 120");
        $("#chart-pr").innerHTML = '<text class="axis-label" x="320" y="62" text-anchor="middle">暂无有效 PR 曲线</text>';
        $("#ev-pr-meta").textContent = "样本不足";
      }

      const pairs = Object.entries(evaluation.pair_metrics || {});
      $("#ev-pairs").innerHTML = pairs.length ? `<thead><tr><th>组合</th><th>支持数</th><th>Precision</th><th>Recall</th><th>F1</th></tr></thead><tbody>${pairs.map(([pair, value]) => `<tr><td class="mono">${pair}</td><td>${value.support ?? "—"}</td><td>${evalNumber(value.precision)}</td><td>${evalNumber(value.recall)}</td><td>${evalNumber(value.f1)}</td></tr>`).join("")}</tbody>` : `<tbody><tr><td>暂无组合指标</td></tr></tbody>`;

      const folds = evaluation.fold_metrics || [];
      $("#ev-folds").innerHTML = folds.length ? `<thead><tr><th>折</th><th>核心 Macro-F1</th><th>全标签 Macro-F1</th><th>Micro-F1</th></tr></thead><tbody>${folds.map(row => `<tr><td>${row.fold}</td><td>${evalNumber(row.core_macro_f1)}</td><td>${evalNumber(row.macro_f1)}</td><td>${evalNumber(row.micro_f1)}</td></tr>`).join("")}<tr class="summary-row"><td>均值 / 标准差</td><td>${evalNumber(evaluation.fold_core_macro_f1_mean)} / ${evalNumber(evaluation.fold_core_macro_f1_std)}</td><td colspan="2">${evaluation.core_macro_f1_ci ? `${evalNumber(evaluation.core_macro_f1_ci.lower)}–${evalNumber(evaluation.core_macro_f1_ci.upper)} CI` : "—"}</td></tr></tbody>` : `<tbody><tr><td>暂无折间指标</td></tr></tbody>`;
      $("#ev-fold-meta").textContent = folds.length ? `${folds.length} 折` : "当前部署无折间记录";

      const stability = evaluation.threshold_stability || {};
      const stableIds = Object.keys(stability).sort((a, b) => Number(a) - Number(b));
      $("#ev-thresholds").innerHTML = stableIds.length ? `<thead><tr><th>类型</th><th>原始</th><th>Bootstrap 中位数</th><th>P05</th><th>P95</th><th>最终阈值</th><th>正样本</th></tr></thead><tbody>${stableIds.map(id => { const row = stability[id]; return `<tr><td>${escapeHtml(names[id] || `类型 ${id}`)}</td><td>${evalNumber(row.raw, 2)}</td><td>${evalNumber(row.bootstrap_median, 2)}</td><td>${evalNumber(row.bootstrap_p05, 2)}</td><td>${evalNumber(row.bootstrap_p95, 2)}</td><td>${evalNumber(row.calibrated ?? evaluation.thresholds?.[id], 2)}</td><td>${row.positive_support ?? "—"}</td></tr>`; }).join("")}</tbody>` : `<tbody><tr><td>当前部署没有阈值稳定性记录</td></tr></tbody>`;
    } else {
      $("#ev-metrics").innerHTML = metricHtml([["评估结果", "不可用"]]);
      ["#ev-perlabel", "#ev-confusion", "#ev-pairs", "#ev-folds", "#ev-thresholds"].forEach(selector => evalEmpty(selector));
      $("#ev-split-meta").textContent = "";
    }

  } catch (error) {
    $("#ev-metrics").innerHTML = metricHtml([["评估结果", "不可用"]]);
    ["#ev-perlabel", "#ev-confusion", "#ev-pairs", "#ev-folds", "#ev-thresholds"].forEach(selector => evalEmpty(selector));
  }
}

/* ---------- 单图诊断 ---------- */
function typeName(fullNames, id) { return fullNames[id] || ("类型 " + id); }

async function loadDiagGrid(reset = false) {
  if (reset) { state.diag.offset = 0; state.diag.selected = null; state.diag.loaded = true; }
  const params = new URLSearchParams({ split: state.diag.split, offset: state.diag.offset, limit: 24, q: $("#diag-search").value });
  const data = await api("/api/dataset/images?" + params);
  const fullNames = (state.model?.type_names_full) || {};
  const grid = $("#diag-grid");
  if (state.diag.offset === 0) grid.innerHTML = "";
  data.items.forEach(item => {
    const tile = document.createElement("div");
    tile.className = "tile" + (item.image_name === state.diag.selected ? " selected" : "");
    tile.dataset.name = item.image_name;
    const types = (item.types || []).map(t => typeName(fullNames, String(t.type_id))).join("、");
    tile.innerHTML = `<img src="/api/dataset/image/${encodeURIComponent(item.image_name)}" loading="lazy" alt=""><div class="cap"><span class="nm">${escapeHtml(item.image_name)}</span><span class="tg">${escapeHtml(types)}</span></div>`;
    tile.addEventListener("click", () => {
      grid.querySelectorAll(".tile").forEach(t => t.classList.toggle("selected", t.dataset.name === item.image_name));
      state.diag.selected = item.image_name;
      diagnoseDatasetImage(item.image_name);
    });
    grid.appendChild(tile);
  });
  state.diag.offset += data.items.length;
  $("#diag-more").disabled = state.diag.offset >= data.total;
  if (!data.items.length && state.diag.offset === 0) grid.innerHTML = '<div class="empty">该集合没有匹配的图像</div>';
}

async function diagnoseDatasetImage(name) {
  $("#diag-result").innerHTML = '<div class="empty">诊断中…</div>';
  try {
    const data = await api("/api/diagnose/batch", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ image_names: [name] }) });
    renderResult(data.results[0]);
  } catch (error) {
    $("#diag-result").innerHTML = `<div class="empty">${escapeHtml(error.message)}</div>`;
  }
}

function probRows(candidates, thresholds, fullNames) {
  return candidates.map(item => {
    const threshold = Number(thresholds[String(item.type_id)] ?? .5);
    const detected = item.confidence >= threshold;
    const cls = detected ? "detected" : "dim";
    return `<div class="prob-row ${cls}"><span class="nm">${escapeHtml(typeName(fullNames, String(item.type_id)))}</span><div class="prob-track"><div class="prob-fill" style="width:${Math.max(0, Math.min(100, item.confidence * 100)).toFixed(1)}%"></div></div><span class="pct">${(item.confidence * 100).toFixed(1)}%<small>/${threshold.toFixed(2)}</small></span></div>`;
  }).join("");
}

function similarRows(cases, fullNames) {
  if (!cases || !cases.length) return '<div class="prob-row dim"><span class="nm">无相似案例</span></div>';
  return cases.map(item => {
    const labels = (item.labels || []).map(id => typeName(fullNames, String(id))).join("、");
    return `<div class="prob-row dim"><span class="nm">${escapeHtml(item.image_name)} · ${escapeHtml(labels)}</span><div class="prob-track"><div class="prob-fill" style="width:${Math.max(0, Math.min(100, item.similarity * 100)).toFixed(1)}%"></div></div><span class="pct">${(item.similarity * 100).toFixed(1)}%</span></div>`;
  }).join("");
}

function renderResult(result) {
  const model = state.model || {};
  const fullNames = model.type_names_full || {};
  const thresholds = result.thresholds || model.thresholds || {};
  if (result.error) {
    $("#diag-result").innerHTML = `<div class="empty">${escapeHtml(result.error)}</div>`;
    return;
  }
  const statusMeta = STATUS_META[result.status] || [result.status, "gray", ""];
  const detected = (result.predicted_types || []).map(id => typeName(fullNames, String(id))).join("、") || "无检出";
  const chips = (result.predicted_types || []).length
    ? (result.predicted_types).map(id => `<span class="chip green">${escapeHtml(typeName(fullNames, String(id)))}</span>`).join(" ")
    : '<span class="chip gold">无检出</span>';
  const trueLabels = (result.true_labels || []).map(t => escapeHtml(t.name)).join("、") || "—";
  const probHtml = probRows(result.top5_candidates || [], thresholds, fullNames);
  const similarHtml = similarRows(result.top5_similar_cases, fullNames);
  $("#diag-result").innerHTML = `
    <div class="result-head">
      <div class="badge-row">
        ${badgeHtml(result.status)}
        <span class="flag ${result.review_required ? "warn" : "ok"}">${result.review_required ? "需人工复核" : "可直接采用"}</span>
      </div>
      <div class="result-class-name">${chips}</div>
      <div class="flag">${escapeHtml(result.image_name)} · ${escapeHtml(statusMeta[2])}</div>
    </div>
    <div class="metric-band" style="padding:6px 0 14px;margin-bottom:0;">
      ${metricHtml([
        ["检出类型", detected],
        ["真实标签", trueLabels],
        ["最高概率", (result.confidence * 100).toFixed(1) + "%"],
        ["决策裕量", result.decision_margin == null ? "—" : Number(result.decision_margin).toFixed(3)],
      ])}
    </div>
    <div class="prob-list"><h3>类型概率（右侧为该类型阈值）</h3>${probHtml}</div>
    <div class="similar-list"><h3>Top-5 相似案例</h3>${similarHtml}</div>`;
}

async function diagnoseUpload(file) {
  const form = new FormData();
  form.append("file", file);
  $("#diag-result").innerHTML = '<div class="empty">诊断中…</div>';
  try {
    renderResult(await api("/api/diagnose/upload", { method: "POST", body: form }));
  } catch (error) {
    $("#diag-result").innerHTML = `<div class="empty">${escapeHtml(error.message)}</div>`;
  }
}

/* ---------- 批量诊断 ---------- */
async function runBatchDataset() {
  const split = $("#batch-split").value;
  const count = Math.min(parseInt($("#batch-count").value, 10) || 1, 200);
  const seed = parseInt($("#batch-seed").value, 10) || 0;
  const data = await api(`/api/dataset/images?split=${encodeURIComponent(split)}&sample_count=${count}&sample_seed=${seed}`);
  await runBatchNames(data.items.map(item => item.image_name), "数据集抽样");
}

async function runBatchUpload() {
  const files = state.batch.files;
  if (!files.length) return;
  const form = new FormData();
  files.forEach(file => form.append("files", file));
  const data = await api("/api/diagnose/batch-upload", { method: "POST", body: form });
  Object.values(state.batch.previewUrls).forEach(url => URL.revokeObjectURL(url));
  state.batch.previewUrls = {};
  data.results.forEach((result, index) => {
    if (files[index]) state.batch.previewUrls[result.image_name] = URL.createObjectURL(files[index]);
  });
  state.batch.results = data.results;
  state.batch.source = "上传文件";
  renderBatchResults();
}

async function runBatchNames(names, source) {
  const data = await api("/api/diagnose/batch", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ image_names: names }) });
  state.batch.results = data.results;
  state.batch.source = source;
  renderBatchResults();
}

function filteredBatchResults() {
  const q = $("#batch-filter").value.toLowerCase();
  const typeFilter = $("#batch-type-filter").value;
  const reviewOnly = $("#batch-review-only").checked;
  return (state.batch.results || []).filter(item => {
    if (item.error) return !q || item.image_name.toLowerCase().includes(q);
    if (q && !item.image_name.toLowerCase().includes(q)) return false;
    if (typeFilter && !(item.predicted_types || []).includes(Number(typeFilter))) return false;
    if (reviewOnly && !item.review_required) return false;
    return true;
  });
}

function renderBatchResults() {
  const results = state.batch.results;
  if (!results) return;
  $("#batch-results").classList.remove("hidden");
  const model = state.model || {};
  const fullNames = model.type_names_full || {};

  const previousType = $("#batch-type-filter").value;
  const types = Object.keys(model.type_names || {}).map(Number).sort((a, b) => a - b);
  $("#batch-type-filter").innerHTML = '<option value="">全部类型</option>'
    + [...types].sort((a, b) => a - b).map(id => `<option value="${id}">${escapeHtml(typeName(fullNames, String(id)))}</option>`).join("");
  if (types.includes(Number(previousType))) $("#batch-type-filter").value = previousType;

  const filtered = filteredBatchResults();
  const reviewed = filtered.filter(item => item.review_required).length;
  const confs = filtered.filter(item => !item.error).map(item => item.confidence);
  const avgConf = confs.length ? (confs.reduce((a, b) => a + b, 0) / confs.length * 100).toFixed(1) + "%" : "—";
  $("#batch-summary").innerHTML = metricHtml([
    ["批量总数", results.length],
    ["筛选结果", filtered.length],
    ["需复核", reviewed],
    ["平均置信度", avgConf],
  ]);
  $("#batch-filtered-info").textContent = `显示 ${filtered.length} / ${results.length}`;

  const rows = filtered.map(item => {
    const isUpload = state.batch.source === "上传文件";
    const thumbUrl = item.error ? "" : isUpload ? (state.batch.previewUrls[item.image_name] || "") : `/api/dataset/image/${encodeURIComponent(item.image_name)}`;
    const thumb = thumbUrl
      ? `<button class="thumb-button" data-name="${escapeHtml(item.image_name)}" data-preview="${escapeHtml(thumbUrl)}" title="点击放大"><img class="row-thumb" src="${escapeHtml(thumbUrl)}" loading="lazy" alt=""></button>`
      : '<span class="thumb-button image-load-failed" title="上传图无法预览"></span>';
    const name = `<span class="cell-name">${escapeHtml(item.image_name)}</span>`;
    if (item.error) {
      return `<tr><td class="thumb-cell">${thumb}${name}</td><td colspan="4">${escapeHtml(item.error)}</td></tr>`;
    }
    const predicted = (item.predicted_types || []).length
      ? item.predicted_types.map(id => escapeHtml(typeName(fullNames, String(id)))).join("<br>")
      : "无检出";
    const truth = (item.true_labels || []).length
      ? item.true_labels.map(t => escapeHtml(t.name)).join("<br>")
      : "—";
    return `<tr>
      <td class="thumb-cell">${thumb}${name}</td>
      <td>${predicted}</td>
      <td>${truth}</td>
      <td class="mono">${(item.confidence * 100).toFixed(1)}%</td>
      <td>${badgeHtml(item.status)}</td>
    </tr>`;
  }).join("");

  $("#batch-table").innerHTML = `<thead><tr><th>图片</th><th>预测类型</th><th>真实标签</th><th>置信度</th><th>状态</th></tr></thead><tbody>${rows || '<tr><td colspan="5"><div class="empty">无匹配结果</div></td></tr>'}</tbody>`;
  document.querySelectorAll("#batch-table .thumb-button[data-name]").forEach(button => {
      button.addEventListener("click", () => openImageDialog(button.dataset.name, "原图预览", button.dataset.preview));
  });
}

function exportBatchCsv() {
  const model = state.model || {};
  const fullNames = model.type_names_full || {};
  const rows = filteredBatchResults().map(item => {
    const predicted = (item.predicted_types || []).map(id => typeName(fullNames, String(id))).join(",");
    const truth = (item.true_labels || []).map(t => t.name).join(",");
    return [item.image_name, predicted, truth, item.status, item.confidence == null ? "" : item.confidence.toFixed(4), item.review_required ? "1" : "0"].map(csvEscape).join(",");
  });
  const csv = "\uFEFF" + ["image_name,predicted_types,true_labels,status,confidence,review_required"].concat(rows).join("\r\n");
  const blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = "batch_predictions.csv";
  link.click();
  URL.revokeObjectURL(link.href);
}

function csvEscape(value) { return `"${String(value ?? "").replace(/"/g, '""')}"`; }

function openImageDialog(name, subtitle, sourceUrl = "") {
  $("#batch-image-title").textContent = name;
  $("#batch-image-subtitle").textContent = subtitle || "原图预览";
  $("#batch-image-error").classList.add("hidden");
  const img = $("#batch-image-preview");
  img.onerror = () => $("#batch-image-error").classList.remove("hidden");
  img.src = sourceUrl || ("/api/dataset/image/" + encodeURIComponent(name));
  $("#batch-image-dialog").showModal();
}

/* ---------- 事件绑定 ---------- */
document.querySelectorAll(".nav-btn").forEach(button => button.addEventListener("click", () => {
  switchView(button.dataset.view);
  if (button.dataset.view === "evaluation") loadEvaluation();
}));

document.querySelectorAll("#diag-mode .seg-btn").forEach(button => button.addEventListener("click", () => {
  document.querySelectorAll("#diag-mode .seg-btn").forEach(x => x.classList.toggle("active", x === button));
  $("#diag-source-dataset").classList.toggle("hidden", button.dataset.mode !== "dataset");
  $("#diag-source-upload").classList.toggle("hidden", button.dataset.mode !== "upload");
  if (button.dataset.mode === "dataset" && !state.diag.loaded) loadDiagGrid(true);
}));

$("#diag-split").addEventListener("change", event => { state.diag.split = event.target.value; loadDiagGrid(true); });
$("#diag-search").addEventListener("change", () => loadDiagGrid(true));
$("#diag-more").addEventListener("click", () => loadDiagGrid());

$("#diag-file").addEventListener("change", event => {
  const file = event.target.files[0];
  if (!file) return;
  const url = URL.createObjectURL(file);
  $("#diag-preview").classList.remove("hidden");
  $("#diag-preview").innerHTML = `<img src="${url}" alt="待诊断图像"><div class="preview-meta">${escapeHtml(file.name)}</div>`;
  diagnoseUpload(file);
});

const dropzone = $("#diag-dropzone");
dropzone.addEventListener("dragover", event => { event.preventDefault(); dropzone.classList.add("dragover"); });
dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
dropzone.addEventListener("drop", event => {
  event.preventDefault();
  dropzone.classList.remove("dragover");
  const file = event.dataTransfer.files[0];
  if (file) { $("#diag-file").files = event.dataTransfer.files; $("#diag-file").dispatchEvent(new Event("change")); }
});

document.querySelectorAll("#batch-mode .seg-btn").forEach(button => button.addEventListener("click", () => {
  document.querySelectorAll("#batch-mode .seg-btn").forEach(x => x.classList.toggle("active", x === button));
  state.batch.mode = button.dataset.mode;
  $("#batch-config-dataset").classList.toggle("hidden", button.dataset.mode !== "dataset");
  $("#batch-config-upload").classList.toggle("hidden", button.dataset.mode !== "upload");
  $("#btn-batch-run").disabled = button.dataset.mode === "upload" && !state.batch.files.length;
}));

$("#batch-file").addEventListener("change", event => {
  state.batch.files = [...event.target.files];
  $("#batch-file-count").textContent = state.batch.files.length ? `已选 ${state.batch.files.length} 张` : "";
  $("#btn-batch-run").disabled = !state.batch.files.length;
});

$("#batch-split").addEventListener("change", event => { state.batch.split = event.target.value; });

const batchDropzone = $("#batch-dropzone");
batchDropzone.addEventListener("dragover", event => { event.preventDefault(); batchDropzone.classList.add("dragover"); });
batchDropzone.addEventListener("dragleave", () => batchDropzone.classList.remove("dragover"));
batchDropzone.addEventListener("drop", event => {
  event.preventDefault();
  batchDropzone.classList.remove("dragover");
  $("#batch-file").files = event.dataTransfer.files;
  $("#batch-file").dispatchEvent(new Event("change"));
});

$("#btn-batch-run").addEventListener("click", async () => {
  $("#btn-batch-run").disabled = true;
  try {
    if (state.batch.mode === "dataset") await runBatchDataset();
    else await runBatchUpload();
  } catch (error) {
    $("#batch-results").classList.remove("hidden");
    $("#batch-summary").innerHTML = metricHtml([["错误", escapeHtml(error.message)]]);
  } finally {
    $("#btn-batch-run").disabled = false;
  }
});

$("#batch-filter").addEventListener("input", renderBatchResults);
$("#batch-type-filter").addEventListener("change", renderBatchResults);
$("#batch-review-only").addEventListener("change", renderBatchResults);
$("#btn-batch-export").addEventListener("click", exportBatchCsv);

$("#batch-image-close").addEventListener("click", () => $("#batch-image-dialog").close());
$("#batch-image-dialog").addEventListener("click", event => { if (event.target === $("#batch-image-dialog")) $("#batch-image-dialog").close(); });

loadOverview();
