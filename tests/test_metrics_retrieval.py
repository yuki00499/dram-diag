import numpy as np

from dram_diag.metrics import stratified_bootstrap
from dram_diag.retrieval import RetrievalIndex


def test_bootstrap_is_reproducible():
    truth = np.array([0, 0, 1, 1])
    values = np.array([0, 1, 1, 1])
    metric = lambda y, x: np.mean(y == x)
    assert stratified_bootstrap(truth, values, metric, 50, 7) == stratified_bootstrap(truth, values, metric, 50, 7)


def test_retrieval_excludes_query_itself():
    records = [{"image_name": "a.jpg", "defect_id": 1}, {"image_name": "b.jpg", "defect_id": 1}]
    index = RetrievalIndex(np.array([[1., 0.], [.9, .1]]), records)
    assert index.search(np.array([1., 0.]), 1, "a.jpg")[0]["image_name"] == "b.jpg"
