const $ = (selector) => document.querySelector(selector);
const state = { model: null, selected: new Set(), offset: 0, imageUrl: null };
const escapeHtml = (value) => String(value ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));

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
}

async function loadOverview() {
  try {
    const model = state.model = await api("/api/model/info");
    $("#ov-ready").textContent = model.ready ? "就绪" : "未部署";
    $("#ov-ready").className = "badge " + (model.ready ? "ok" : "warn");
    $("#status-dot").className = "status-dot " + (model.ready ? "ok" : "warn");
    $("#status-text").textContent = model.ready ? "就绪" : "模型未部署";
    const cards = [
      ["协议", model.protocol_version], ["部署模式", model.mode], ["架构", model.architecture],
      ["分类范围", model.num_classes + " 类"], ["覆盖样本", `${model.classification_samples} / ${model.total_samples}`],
      ["未知拒识", model.unknown_rejection_enabled ? "已启用" : "未启用"],
      ["验证 Macro-F1", model.best_val_macro_f1 == null ? "—" : (model.best_val_macro_f1 * 100).toFixed(1) + "%"],
      ["设备", model.device]
    ];
    $("#ov-params").innerHTML = cards.map(([key, value]) => `<div class="stat-card"><div class="k">${escapeHtml(key)}</div><div class="v small">${escapeHtml(value)}</div></div>`).join("");
    $("#footer-model").textContent = "模型: " + (model.model_version || "未部署");
    $("#footer-classes").textContent = model.num_classes + " 类";
    $("#footer-device").textContent = "协议: " + model.protocol_version;
    const history = await api("/api/model/history");
    if (history.length) {
      lineChart($("#chart-loss"), [{ name: "train", color: "#1B66F5", values: history.map(x => x.train_loss) }, { name: "validation", color: "#D97706", values: history.map(x => x.validation_loss) }], { yMin: 0 });
      lineChart($("#chart-acc"), [{ name: "train", color: "#9CA3AF", values: history.map(x => x.train_accuracy) }, { name: "validation", color: "#1B66F5", values: history.map(x => x.validation_accuracy) }], { yMin: 0, yMax: 1 });
      $("#ov-epochs").textContent = history.length + " epochs";
    }
    const dist = await api("/api/model/class-distribution");
    if (dist.task === "multilabel") {
      const typeEntries = Object.entries(dist.type_counts).sort((a, b) => Number(a[0]) - Number(b[0]));
      const typeNames = dist.type_names || {};
      $("#batch-defect").innerHTML = '<option value="">全部类型</option>' + typeEntries.map(([id, count]) => `<option value="${id}">${escapeHtml(typeNames[id] || ("类型 " + id))}（${count}）</option>`).join("");
      if (typeEntries.length) barChart($("#chart-dist"), typeEntries.map(([id, value]) => ({ label: typeNames[id] || ("类型 " + id), value, highlight: false })));
      $("#ov-dist-stats").innerHTML = `<div class="dist-tier"><div class="t">缺陷类型</div><div class="n">${dist.types.length} <em>类</em></div></div><div class="dist-tier"><div class="t">训练 / 验证</div><div class="n">${dist.split_sizes.train || 0} <em>/ ${dist.split_sizes.validation || 0}</em></div></div>`;
      return;
    }
    $("#batch-defect").innerHTML = '<option value="">全部缺陷</option>' + Object.keys(dist.counts).sort((a,b) => Number(a)-Number(b)).map(id => `<option value="${id}">缺陷 ${id}</option>`).join("");
    const values = Object.entries(dist.counts).map(([label, value]) => ({ label, value, highlight: !dist.classification_classes.includes(Number(label)) }));
    if (values.length) barChart($("#chart-dist"), values);
    $("#ov-dist-stats").innerHTML = `<div class="dist-tier"><div class="t">分类类</div><div class="n">${dist.classification_classes.length} <em>类</em></div></div><div class="dist-tier"><div class="t">案例库类</div><div class="n">${dist.case_library_classes.length} <em>类</em></div></div><div class="dist-tier"><div class="t">训练 / 验证</div><div class="n">${dist.split_sizes.train || 0} <em>/ ${dist.split_sizes.validation || 0}</em></div></div>`;
  } catch (error) {
    $("#status-text").textContent = "服务不可用";
  }
}

function evidence(value, color = "") {
  return `<div class="top5-track"><div class="top5-fill" style="width:${Math.max(0, Math.min(100, value * 100))}%;${color}"></div></div>`;
}

function renderResult(result) {
  const labels = { known_confident: "可信分类", known_uncertain: "待复核", unknown: "未知", low_quality: "图像质量不足" };
  if (result.predicted_types !== undefined) {
    const chips = result.predicted_types.length
      ? result.predicted_types.map(id => {
          const item = result.top5_candidates.find(c => c.type_id === id);
          return `<span class="result-chip">${escapeHtml(item ? item.name : "类型 " + id)}</span>`;
        }).join(" ")
      : '<span class="result-chip warn">无检出</span>';
    const probs = result.top5_candidates.map(item => `<div class="top5-row"><span class="nm">${escapeHtml(item.name)}</span>${evidence(item.confidence)}<span class="pct">${(item.confidence * 100).toFixed(1)}%</span></div>`).join("");
    const similar = result.top5_similar_cases.map(item => `<div class="top5-row dim"><span class="nm">${escapeHtml(item.image_name)} · 类型 ${escapeHtml(String(item.labels || "").split(",")[0])}</span>${evidence(item.similarity)}<span class="pct">${(item.similarity * 100).toFixed(1)}%</span></div>`).join("");
    $("#diag-result").innerHTML = `<div class="result-head"><div><div class="result-class-name">${chips}</div><div class="result-class-sub">${escapeHtml(result.image_name)} · ${escapeHtml(labels[result.status])}</div><span class="result-flag ${result.review_required ? "warn" : "ok"}">${result.review_required ? "需要人工复核" : "可直接采用"}</span></div></div><div class="top5"><h3>类型概率（阈值 0.5）</h3>${probs}</div><div class="top5"><h3>Top-5 相似案例</h3>${similar}</div>`;
    return;
  }
  const candidates = result.top5_candidates.map(item => `<div class="top5-row"><span class="nm">缺陷 ${item.defect_id}</span>${evidence(item.confidence)}<span class="pct">${(item.confidence * 100).toFixed(1)}%</span></div>`).join("");
  const similar = result.top5_similar_cases.map(item => `<div class="top5-row dim"><span class="nm">${escapeHtml(item.image_name)} · 缺陷 ${item.defect_id}</span>${evidence(item.similarity)}<span class="pct">${(item.similarity * 100).toFixed(1)}%</span></div>`).join("");
  $("#diag-result").innerHTML = `<div class="result-head"><div><div class="result-class-name">${result.predicted_class == null ? "未知缺陷" : "缺陷 " + result.predicted_class}</div><div class="result-class-sub">${escapeHtml(result.image_name)} · ${escapeHtml(labels[result.status])}</div><span class="result-flag ${result.review_required ? "warn" : "ok"}">${result.review_required ? "需要人工复核" : "可直接采用"}</span></div></div><div class="top5"><h3>分类证据</h3><div class="top5-row"><span class="nm">分类概率</span>${evidence(result.confidence)}<span class="pct">${(result.confidence * 100).toFixed(1)}%</span></div><div class="top5-row"><span class="nm">原型相似度</span>${evidence(result.prototype_similarity)}<span class="pct">${(result.prototype_similarity * 100).toFixed(1)}%</span></div><div class="top5-row"><span class="nm">未知分数</span>${evidence(result.unknown_score, "background:#D97706")}<span class="pct">${(result.unknown_score * 100).toFixed(1)}%</span></div></div><div class="top5"><h3>Top-5 分类候选</h3>${candidates}</div><div class="top5"><h3>Top-5 相似案例</h3>${similar}</div>`;
}

async function diagnoseFile(file) {
  const form = new FormData(); form.append("file", file);
  $("#diag-result").innerHTML = '<div class="empty">诊断中…</div>';
  try { renderResult(await api("/api/diagnose/upload", { method: "POST", body: form })); }
  catch (error) { $("#diag-result").innerHTML = `<div class="empty">${escapeHtml(error.message)}</div>`; }
}

async function loadBatch(reset = false) {
  if (reset) { state.offset = 0; $("#batch-list").innerHTML = ""; }
  const params = new URLSearchParams({ offset: state.offset, limit: 40, q: $("#batch-search").value, defect_id: $("#batch-defect").value });
  if (!$("#batch-defect").value) params.delete("defect_id");
  const data = await api("/api/dataset/images?" + params);
  data.items.forEach(item => {
    const label = document.createElement("label"); label.className = "pick-item";
    const typeText = item.types !== undefined
      ? item.types.map(t => t.name).join("、")
      : (item.defect_id != null ? "缺陷 " + item.defect_id : "");
    label.innerHTML = `<input type="checkbox" data-name="${escapeHtml(item.image_name)}" ${state.selected.has(item.image_name) ? "checked" : ""}><img class="pick-thumb" src="/api/dataset/image/${encodeURIComponent(item.image_name)}" loading="lazy" data-name="${escapeHtml(item.image_name)}" alt=""><span class="nm">${escapeHtml(item.image_name)}</span><span class="did">${escapeHtml(typeText)}</span>`;
    label.querySelector("img").addEventListener("click", event => { event.preventDefault(); event.stopPropagation(); openBatchImage(item.image_name); });
    $("#batch-list").appendChild(label);
  });
  state.offset += data.items.length; $("#batch-more").disabled = state.offset >= data.total;
}

async function runBatch() {
  const names = [...state.selected];
  if (!names.length) return;
  try {
    const data = await api("/api/diagnose/batch", { method: "POST", headers: {"Content-Type":"application/json"}, body: JSON.stringify({ image_names: names }) });
    const isMultilabel = data.results[0] && data.results[0].predicted_types !== undefined;
    const head = isMultilabel
      ? `<tr><th>图片</th><th>诊断结果</th><th>真实标签</th><th>置信度</th><th>状态</th></tr>`
      : `<tr><th>图片</th><th>结果</th><th>置信度</th><th>状态</th></tr>`;
    const rows = data.results.map(item => {
      const thumb = `<img class="batch-thumb" src="/api/dataset/image/${encodeURIComponent(item.image_name)}" loading="lazy" data-name="${escapeHtml(item.image_name)}" alt="">`;
      if (item.error) {
        return `<tr><td>${thumb}<span class="cell-name">${escapeHtml(item.image_name)}</span></td><td colspan="${isMultilabel ? 4 : 3}">${escapeHtml(item.error)}</td></tr>`;
      }
      if (isMultilabel) {
        const predicted = item.predicted_types.length
          ? item.predicted_types.map(id => {
              const candidate = item.top5_candidates.find(c => c.type_id === id);
              return escapeHtml(candidate ? candidate.name : "类型" + id);
            }).join("<br>")
          : "无检出";
        const truth = item.true_labels && item.true_labels.length
          ? item.true_labels.map(t => escapeHtml(t.name)).join("<br>")
          : "—";
        const conf = (item.confidence * 100).toFixed(1) + "%";
        return `<tr><td>${thumb}<span class="cell-name">${escapeHtml(item.image_name)}</span></td><td>${predicted}</td><td>${truth}</td><td>${conf}</td><td>${escapeHtml(item.status)}</td></tr>`;
      }
      const conf = (item.confidence * 100).toFixed(1) + "%";
      return `<tr><td>${thumb}<span class="cell-name">${escapeHtml(item.image_name)}</span></td><td>${item.predicted_class ?? "未知"}</td><td>${conf}</td><td>${escapeHtml(item.status)}</td></tr>`;
    }).join("");
    $("#batch-result").innerHTML = `<div class="table-wrap"><table><thead>${head}</thead><tbody>${rows}</tbody></table></div>`;
    document.querySelectorAll(".batch-thumb").forEach(img => {
      img.addEventListener("click", () => openBatchImage(img.dataset.name));
    });
  } catch (error) { $("#batch-result").innerHTML = `<div class="empty">${escapeHtml(error.message)}</div>`; }
}

function openBatchImage(name) {
  $("#batch-image-title").textContent = name;
  $("#batch-image-subtitle").textContent = "原图预览";
  $("#batch-image-error").classList.add("hidden");
  $("#batch-image-preview").onerror = () => $("#batch-image-error").classList.remove("hidden");
  $("#batch-image-preview").src = "/api/dataset/image/" + encodeURIComponent(name);
  $("#batch-image-dialog").showModal();
}

document.querySelectorAll(".nav-btn").forEach(button => button.addEventListener("click", () => switchView(button.dataset.view)));
document.querySelectorAll(".seg-btn").forEach(button => button.addEventListener("click", () => { document.querySelectorAll(".seg-btn").forEach(x => x.classList.toggle("active", x === button)); $("#diag-single").classList.toggle("hidden", button.dataset.mode !== "single"); $("#diag-batch").classList.toggle("hidden", button.dataset.mode !== "batch"); if (button.dataset.mode === "batch" && !state.offset) loadBatch(); }));
$("#file-input").addEventListener("change", event => { const file = event.target.files[0]; if (!file) return; if (state.imageUrl) URL.revokeObjectURL(state.imageUrl); state.imageUrl = URL.createObjectURL(file); $("#diag-preview").innerHTML = `<img src="${state.imageUrl}" alt="待诊断图像">`; $("#diag-preview-meta").textContent = file.name; diagnoseFile(file); });
$("#batch-list").addEventListener("change", event => { if (!event.target.dataset.name) return; event.target.checked ? state.selected.add(event.target.dataset.name) : state.selected.delete(event.target.dataset.name); $("#batch-count").textContent = `已选 ${state.selected.size} 张`; });
$("#batch-more").addEventListener("click", () => loadBatch());
$("#batch-search").addEventListener("change", () => loadBatch(true));
$("#batch-defect").addEventListener("change", () => loadBatch(true));
$("#btn-batch-run").addEventListener("click", runBatch);
$("#batch-select-all").addEventListener("change", event => document.querySelectorAll("#batch-list input").forEach(box => { box.checked = event.target.checked; box.dispatchEvent(new Event("change", { bubbles: true })); }));
$("#batch-image-close").addEventListener("click", () => $("#batch-image-dialog").close());
$("#batch-image-dialog").addEventListener("click", event => { if (event.target === $("#batch-image-dialog")) $("#batch-image-dialog").close(); });
loadOverview();
