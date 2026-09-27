const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

test("pending review class filter counts multi-label images and skips reviewed images", () => {
  const elements = new Map();
  const element = id => {
    if (!elements.has(id)) elements.set(id, {checked: false, hidden: false, disabled: false, textContent: "", addEventListener() {}});
    return elements.get(id);
  };
  element("canvas").getContext = () => ({});
  element("filter-class").value = "all";
  element("filter-class").options = [
    {value: "all", dataset: {label: "全部类别"}},
    {value: "none", dataset: {label: "无缺陷框"}},
    {value: "0", dataset: {label: "近圆形颗粒"}},
    {value: "1", dataset: {label: "细长颗粒"}},
  ];
  const context = vm.createContext({
    document: {getElementById: element, addEventListener() {}},
    window: {addEventListener() {}},
    fetch: () => new Promise(() => {}),
  });
  const script = fs.readFileSync(path.join(__dirname, "..", "annotation", "app.js"), "utf8");
  vm.runInContext(script, context);
  vm.runInContext(`
    config = {images: ["a.jpg", "b.jpg", "c.jpg", "d.jpg"], total: 4, review_sample: ["a.jpg", "b.jpg"]};
    payload = {images: {
      "a.jpg": {review: {status: "unreviewed"}, objects: [{class_id: 0}, {class_id: 1}]},
      "b.jpg": {review: {status: "reviewed"}, objects: [{class_id: 1}]},
      "c.jpg": {review: {status: "primary_complete"}, objects: []},
      "d.jpg": {review: {status: "disputed"}, objects: [{class_id: 0}]},
    }};
    position = 0;
    image = {complete: true};
    renderAll = () => {};
    loadImage = () => {};
  `, context);

  element("only-pending-review").checked = true;
  element("filter-class").value = "0";
  assert.deepEqual(Array.from(vm.runInContext("updateFilterUi()", context)), [0, 3]);
  assert.equal(element("filter-class").options[2].textContent, "近圆形颗粒（2 张）");
  assert.equal(element("filter-class").options[3].textContent, "细长颗粒（1 张）");
  vm.runInContext("navigate(1)", context);
  assert.equal(vm.runInContext("position", context), 3);

  vm.runInContext('payload.images["d.jpg"].review.status = "reviewed"; refreshVisibleImage()', context);
  assert.equal(vm.runInContext("position", context), 0);
  assert.deepEqual(Array.from(vm.runInContext("filteredIndices()", context)), [0]);

  element("filter-class").value = "none";
  assert.deepEqual(Array.from(vm.runInContext("filteredIndices()", context)), [2]);
  element("only-review-sample").checked = true;
  vm.runInContext("refreshVisibleImage()", context);
  assert.equal(element("filter-empty").hidden, false);
  assert.equal(element("prev").disabled, true);
});
