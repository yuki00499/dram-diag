"use strict";

const $ = id => document.getElementById(id);
const STORAGE_KEY = "dram_det_v3_annotations";
const COLORS = ["#ff6b6b", "#ffad42", "#ffe66d", "#65d18a", "#42d4c7", "#4aa3ff", "#a98bff", "#f17bd3"];
let config, payload, image, position = 0, selectedClass = null, ignoreMode = false;
let selected = null, pointerAction = null, undoStack = [], redoStack = [], blindReview = false;
const canvas = $("canvas"), ctx = canvas.getContext("2d");

function clone(value) { return JSON.parse(JSON.stringify(value)); }
function currentName() { return config.images[position]; }
function classInfo(id) { return config.taxonomy.object_classes.find(item => Number(item.id) === Number(id)); }
function colorFor(id) { return COLORS[Number(id) % COLORS.length]; }
function taxonomyHash() { return config.taxonomy.taxonomy_sha256; }

function emptyRecord(width, height) {
  return { width, height, objects: [], quality_attributes: [], usability: "review", ignore_regions: [],
    review: { status: "unreviewed", note: "", secondary_objects: [], secondary_ignore_regions: [] } };
}

function record() {
  const name = currentName();
  if (!payload.images[name]) payload.images[name] = emptyRecord(image?.naturalWidth || 480, image?.naturalHeight || 320);
  return payload.images[name];
}
function ensureReview(){ const row=record(); row.review=row.review||{status:"unreviewed",note:""}; row.review.secondary_objects=row.review.secondary_objects||[]; row.review.secondary_ignore_regions=row.review.secondary_ignore_regions||[]; return row.review; }
function activeObjects(){ return blindReview ? ensureReview().secondary_objects : record().objects; }
function activeIgnores(){ return blindReview ? ensureReview().secondary_ignore_regions : record().ignore_regions; }

function persistLocal() {
  try { localStorage.setItem(STORAGE_KEY, JSON.stringify(payload)); } catch (_) {}
}

function commit(before) {
  const after = clone(record());
  if (JSON.stringify(before) === JSON.stringify(after)) return;
  undoStack.push({ name: currentName(), before, after });
  if (undoStack.length > 200) undoStack.shift();
  redoStack = [];
  persistLocal();
  renderAll();
}

function applyHistory(entry, side) {
  payload.images[entry.name] = clone(entry[side]);
  if (entry.name === currentName()) { selected = null; renderAll(); }
  persistLocal();
}

function undo() { const item = undoStack.pop(); if (!item) return; applyHistory(item, "before"); redoStack.push(item); }
function redo() { const item = redoStack.pop(); if (!item) return; applyHistory(item, "after"); undoStack.push(item); }

function buildControls() {
  const classList = $("classes");
  classList.innerHTML = "";
  if (!config.taxonomy.object_classes.length) {
    classList.innerHTML = '<div class="danger">类别表仍为空：请先完成试标类别定义。</div>';
  }
  config.taxonomy.object_classes.forEach(item => {
    const button = document.createElement("button");
    button.className = "class-btn";
    button.innerHTML = `<span class="swatch" style="background:${colorFor(item.id)}"></span><strong>${item.id}</strong><span>${escapeHtml(item.name_zh)}<small> ${escapeHtml(item.name_en)}</small></span>`;
    button.title = `${item.definition || ""}\n边界：${item.boundary_rule || ""}`;
    button.onclick = () => { selectedClass = Number(item.id); ignoreMode = false; selected = null; renderAll(); };
    button.dataset.classId = item.id;
    classList.appendChild(button);
  });
  const qualities = $("qualities");
  qualities.innerHTML = "";
  config.taxonomy.quality_attributes.forEach(item => {
    const label = document.createElement("label");
    label.innerHTML = `<input type="checkbox" value="${escapeHtml(item.id)}"> ${escapeHtml(item.name_zh)} <small>${escapeHtml(item.name_en)}</small>`;
    label.querySelector("input").onchange = event => {
      const before = clone(record()), values = new Set(record().quality_attributes);
      event.target.checked ? values.add(item.id) : values.delete(item.id);
      record().quality_attributes = [...values].sort(); commit(before);
    };
    qualities.appendChild(label);
  });
  const usability = $("usability"); usability.innerHTML = "";
  const labels = { usable: "可用", review: "待复核", unusable: "不可用" };
  config.taxonomy.usability_states.forEach(value => {
    const button = document.createElement("button"); button.textContent = labels[value] || value; button.dataset.value = value;
    button.onclick = () => { const before = clone(record()); record().usability = value; commit(before); };
    usability.appendChild(button);
  });
}

function loadImage() {
  selected = null; pointerAction = null;
  image = new Image();
  image.onload = () => {
    const name = currentName();
    if (!payload.images[name]) payload.images[name] = emptyRecord(image.naturalWidth, image.naturalHeight);
    canvas.width = image.naturalWidth; canvas.height = image.naturalHeight;
    renderAll();
  };
  image.src = `/api/annotation/image/${encodeURIComponent(currentName())}`;
  $("image-name").textContent = currentName();
  $("position").textContent = `${position + 1} / ${config.total}`;
}

function renderCanvas() {
  if (!image?.complete) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height); ctx.drawImage(image, 0, 0);
  activeObjects().forEach((obj, index) => drawBox(obj.bbox_xyxy, colorFor(obj.class_id),
    $("show-labels").checked ? `${obj.class_id} ${classInfo(obj.class_id)?.name_zh || ""}` : "", selected?.kind === "object" && selected.index === index));
  activeIgnores().forEach((region, index) => drawBox(region.bbox_xyxy, "#ffbf4a",
    $("show-labels").checked ? `IGNORE ${region.reason || "争议"}` : "", selected?.kind === "ignore" && selected.index === index, true));
  if (pointerAction?.kind === "draw") drawBox(pointerAction.box, pointerAction.ignore ? "#ffbf4a" : colorFor(selectedClass), "", true, pointerAction.ignore);
}

function drawBox(box, color, label, active, dashed=false) {
  const [x1,y1,x2,y2] = box; ctx.save(); ctx.strokeStyle = color; ctx.lineWidth = active ? 3 : 2;
  if (dashed) ctx.setLineDash([7,5]); ctx.strokeRect(x1,y1,x2-x1,y2-y1); ctx.setLineDash([]);
  if (label) { ctx.font = "13px system-ui"; const width = ctx.measureText(label).width + 10; ctx.fillStyle = color; ctx.fillRect(x1, Math.max(0,y1-20), width, 20); ctx.fillStyle="#071018"; ctx.fillText(label,x1+5,Math.max(14,y1-5)); }
  if (active) [[x1,y1],[x2,y1],[x1,y2],[x2,y2]].forEach(([x,y]) => { ctx.fillStyle="#fff"; ctx.fillRect(x-4,y-4,8,8); });
  ctx.restore();
}

function renderAll() {
  if (!image?.complete) return;
  document.querySelectorAll(".class-btn").forEach(button => button.classList.toggle("selected", !ignoreMode && Number(button.dataset.classId) === selectedClass));
  $("ignore-mode").classList.toggle("selected", ignoreMode);
  $("mode-label").textContent = ignoreMode ? "拖动创建争议/忽略区域" : selectedClass == null ? "选择缺陷类别后拖动画框" : `当前类别：${classInfo(selectedClass)?.name_zh || selectedClass}`;
  document.querySelectorAll("#qualities input").forEach(input => input.checked = record().quality_attributes.includes(input.value));
  document.querySelectorAll("#usability button").forEach(button => button.classList.toggle("selected", button.dataset.value === record().usability));
  $("review-status").value = record().review?.status || "unreviewed"; $("review-note").value = record().review?.note || "";
  $("blind-review").textContent = blindReview ? "退出盲复核（查看主标）" : "进入盲复核";
  $("blind-review").classList.toggle("selected", blindReview);
  renderObjects(); renderStats(); renderCanvas();
}

function renderObjects() {
  const target = $("objects"); target.innerHTML = "";
  activeObjects().forEach((obj,index) => {
    const row = document.createElement("div"); row.className = "object-row" + (selected?.kind === "object" && selected.index === index ? " active" : "");
    row.innerHTML = `<strong style="color:${colorFor(obj.class_id)}">${obj.class_id} ${escapeHtml(classInfo(obj.class_id)?.name_zh || "")}</strong><br><small>${obj.bbox_xyxy.map(v=>Math.round(v)).join(", ")} · ${escapeHtml(obj.instance_id)}</small>`;
    row.onclick = () => { selected = {kind:"object", index}; renderAll(); }; target.appendChild(row);
  });
  activeIgnores().forEach((region,index) => {
    const row = document.createElement("div"); row.className = "object-row ignore" + (selected?.kind === "ignore" && selected.index === index ? " active" : "");
    row.innerHTML = `<strong>IGNORE</strong><br><small>${region.bbox_xyxy.map(v=>Math.round(v)).join(", ")} · ${escapeHtml(region.reason || "争议")}</small>`;
    row.onclick = () => { selected = {kind:"ignore", index}; renderAll(); }; target.appendChild(row);
  });
  if (!target.children.length) target.textContent = "尚无缺陷框";
}

function renderStats() {
  const records = Object.values(payload.images), objects = records.reduce((sum,row)=>sum+(row.objects?.length||0),0);
  const reviewed = records.filter(row => ["reviewed","primary_complete"].includes(row.review?.status)).length;
  const secondary = records.reduce((sum,row)=>sum+(row.review?.secondary_objects?.length||0),0);
  $("stats").innerHTML = `<span>已打开/总数</span><strong>${records.length}/${config.total}</strong><span>主标框</span><strong>${objects}</strong><span>盲复核框</span><strong>${secondary}</strong><span>已审/主标完成</span><strong>${reviewed}</strong><span>争议区域</span><strong>${records.reduce((s,r)=>s+(r.ignore_regions?.length||0),0)}</strong>`;
}

function point(event) { const rect=canvas.getBoundingClientRect(); return [
  Math.max(0,Math.min(canvas.width,(event.clientX-rect.left)*canvas.width/rect.width)),
  Math.max(0,Math.min(canvas.height,(event.clientY-rect.top)*canvas.height/rect.height))]; }
function selectedShape() { if (!selected) return null; return selected.kind === "object" ? activeObjects()[selected.index] : activeIgnores()[selected.index]; }
function handleAt(box,x,y) { const d=9*canvas.width/canvas.getBoundingClientRect().width; const pts=[[box[0],box[1],"nw"],[box[2],box[1],"ne"],[box[0],box[3],"sw"],[box[2],box[3],"se"]]; return pts.find(([px,py])=>Math.abs(px-x)<=d&&Math.abs(py-y)<=d)?.[2]; }
function inside(box,x,y) { return x>=box[0]&&x<=box[2]&&y>=box[1]&&y<=box[3]; }
function hitShape(x,y) {
  for (let i=activeIgnores().length-1;i>=0;i--) if(inside(activeIgnores()[i].bbox_xyxy,x,y)) return {kind:"ignore",index:i};
  for (let i=activeObjects().length-1;i>=0;i--) if(inside(activeObjects()[i].bbox_xyxy,x,y)) return {kind:"object",index:i};
  return null;
}

canvas.addEventListener("pointerdown", event => {
  const [x,y]=point(event), before=clone(record()), shape=selectedShape();
  if (shape) { const handle=handleAt(shape.bbox_xyxy,x,y); if(handle){ pointerAction={kind:"resize",handle,start:[x,y],original:[...shape.bbox_xyxy],before}; canvas.setPointerCapture(event.pointerId); return; } if(inside(shape.bbox_xyxy,x,y)){ pointerAction={kind:"move",start:[x,y],original:[...shape.bbox_xyxy],before}; canvas.setPointerCapture(event.pointerId); return; } }
  const hit=hitShape(x,y); if(hit){ selected=hit; renderAll(); return; }
  if (ignoreMode || selectedClass != null) { selected=null; pointerAction={kind:"draw",start:[x,y],box:[x,y,x,y],ignore:ignoreMode,before}; canvas.setPointerCapture(event.pointerId); }
});
canvas.addEventListener("pointermove", event => {
  if(!pointerAction) return; const [x,y]=point(event);
  if(pointerAction.kind==="draw") pointerAction.box=[Math.min(pointerAction.start[0],x),Math.min(pointerAction.start[1],y),Math.max(pointerAction.start[0],x),Math.max(pointerAction.start[1],y)];
  else { const shape=selectedShape(); if(!shape)return; const b=[...pointerAction.original], dx=x-pointerAction.start[0],dy=y-pointerAction.start[1];
    if(pointerAction.kind==="move"){ const w=b[2]-b[0],h=b[3]-b[1]; b[0]=Math.max(0,Math.min(canvas.width-w,b[0]+dx));b[1]=Math.max(0,Math.min(canvas.height-h,b[1]+dy));b[2]=b[0]+w;b[3]=b[1]+h; }
    else { if(pointerAction.handle.includes("w"))b[0]=Math.min(b[2]-2,Math.max(0,x)); if(pointerAction.handle.includes("e"))b[2]=Math.max(b[0]+2,Math.min(canvas.width,x)); if(pointerAction.handle.includes("n"))b[1]=Math.min(b[3]-2,Math.max(0,y)); if(pointerAction.handle.includes("s"))b[3]=Math.max(b[1]+2,Math.min(canvas.height,y)); } shape.bbox_xyxy=b; }
  renderCanvas();
});
canvas.addEventListener("pointerup", () => {
  if(!pointerAction)return; const action=pointerAction; pointerAction=null;
  if(action.kind==="draw" && action.box[2]-action.box[0]>=3 && action.box[3]-action.box[1]>=3){
    if(action.ignore){ const reason=prompt("争议/忽略原因","边界或类别不确定")||"未说明"; activeIgnores().push({bbox_xyxy:action.box.map(v=>Math.round(v*100)/100),reason,candidate_class_ids:selectedClass==null?[]:[selectedClass]}); selected={kind:"ignore",index:activeIgnores().length-1}; }
    else { activeObjects().push({instance_id:`${blindReview?"review-":""}${PathStem(currentName())}-${Date.now().toString(36)}`,class_id:selectedClass,bbox_xyxy:action.box.map(v=>Math.round(v*100)/100)}); selected={kind:"object",index:activeObjects().length-1}; }
  }
  commit(action.before); renderAll();
});

function PathStem(name){ return name.replace(/\.[^.]+$/,""); }
function removeSelected(){ if(!selected)return; const before=clone(record()); selected.kind==="object"?activeObjects().splice(selected.index,1):activeIgnores().splice(selected.index,1);selected=null;commit(before); }
function navigate(delta){ let next=position; for(let tries=0;tries<config.total;tries++){ next=(next+delta+config.total)%config.total; const name=config.images[next], row=payload.images[name]; const statusOk=!$("only-unreviewed").checked || !row || row.review?.status==="unreviewed"; const sampleOk=!$("only-review-sample").checked || (config.review_sample||[]).includes(name); if(statusOk&&sampleOk) break; } position=next;loadImage(); }
function escapeHtml(value){ const div=document.createElement("div");div.textContent=String(value??"");return div.innerHTML; }

$("prev").onclick=()=>navigate(-1); $("next").onclick=()=>navigate(1); $("undo").onclick=undo; $("redo").onclick=redo;
$("blind-review").onclick=()=>{blindReview=!blindReview;selected=null;pointerAction=null;renderAll();};
$("ignore-mode").onclick=()=>{ignoreMode=!ignoreMode;selected=null;renderAll();}; $("show-labels").onchange=renderCanvas;
$("jump").onkeydown=event=>{if(event.key!=="Enter")return;const raw=event.target.value.trim();let index=/^\d+$/.test(raw)?Number(raw)-1:config.images.indexOf(raw);if(index>=0&&index<config.total){position=index;loadImage();event.target.value="";}};
$("review-status").onchange=event=>{const before=clone(record());record().review=record().review||{};record().review.status=event.target.value;commit(before);};
$("review-note").onchange=event=>{const before=clone(record());record().review=record().review||{};record().review.note=event.target.value;commit(before);};
$("export").onclick=()=>{const blob=new Blob([JSON.stringify(payload,null,2)],{type:"application/json"}),link=document.createElement("a");link.href=URL.createObjectURL(blob);link.download="dram_det_v3.json";link.click();URL.revokeObjectURL(link.href);};
$("import").onclick=()=>$("import-file").click();
$("import-file").onchange=event=>{const file=event.target.files[0];if(!file)return;const reader=new FileReader();reader.onload=()=>{try{const value=JSON.parse(reader.result);if(value.schema_version!==3||value.taxonomy_sha256!==taxonomyHash())throw new Error("协议或类别哈希不匹配");payload=value;persistLocal();loadImage();message(`已导入 ${Object.keys(payload.images||{}).length} 张`);}catch(error){message(error.message,true);}};reader.readAsText(file);};
$("save").onclick=async()=>{try{const response=await fetch("/api/annotation/state",{method:"PUT",headers:{"Content-Type":"application/json"},body:JSON.stringify({payload})});const result=await response.json();if(!response.ok)throw new Error(JSON.stringify(result.detail));message(`已保存，${result.warnings?.length||0} 条警告`);}catch(error){message(error.message,true);}};
function message(value,danger=false){$("messages").textContent=value;$("messages").classList.toggle("danger",danger);}
document.addEventListener("keydown",event=>{if(["INPUT","TEXTAREA","SELECT"].includes(event.target.tagName))return;if(event.ctrlKey&&event.key.toLowerCase()==="z"){event.preventDefault();undo();}else if(event.ctrlKey&&event.key.toLowerCase()==="y"){event.preventDefault();redo();}else if(event.key==="Delete")removeSelected();else if(event.key==="ArrowRight")navigate(1);else if(event.key==="ArrowLeft")navigate(-1);else if(event.key.toLowerCase()==="i"){ignoreMode=!ignoreMode;selected=null;renderAll();}else if(/^\d$/.test(event.key)){const item=config.taxonomy.object_classes[Number(event.key)-1];if(item){selectedClass=Number(item.id);ignoreMode=false;renderAll();}}});
window.addEventListener("resize",renderCanvas);

Promise.all([fetch("/api/annotation/config").then(r=>r.json()),fetch("/api/annotation/state").then(r=>r.json())]).then(([cfg,server])=>{
  config=cfg; const local=localStorage.getItem(STORAGE_KEY); payload=server;
  if(local){try{const candidate=JSON.parse(local);if(candidate.taxonomy_sha256===taxonomyHash()&&Object.keys(candidate.images||{}).length>=Object.keys(server.images||{}).length)payload=candidate;}catch(_) {}}
  payload.images=payload.images||{}; payload.taxonomy_sha256=taxonomyHash();
  $("taxonomy-state").textContent=`taxonomy ${config.taxonomy.status} · ${config.taxonomy.object_classes.length} 类 · ${config.total} 张`;
  buildControls();loadImage();
}).catch(error=>message("初始化失败："+error.message,true));
