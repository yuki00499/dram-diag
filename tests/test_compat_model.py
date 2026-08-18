import pytest
import torch

from dram_diag.compat import validate_checkpoint
from dram_diag.model import DefectClassifier
from dram_diag.training import build_label_graph


def test_model_has_seven_outputs_without_download():
    output = DefectClassifier(7, pretrained=False)(torch.zeros(2, 3, 64, 64))
    assert output["logits"].shape == (2, 7)
    assert output["embedding"].shape == (2, 512)


def test_checkpoint_protocol_mismatch_is_rejected():
    manifest = {"protocol_version": "dram-ml-v2", "manifest_fingerprint": "abc", "types": [2, 4]}
    checkpoint = {"protocol_version": "old", "manifest_fingerprint": "abc", "class_to_idx": {3: 0, 4: 1}}
    with pytest.raises(ValueError):
        validate_checkpoint(checkpoint, manifest)


def test_graph_classifier_keeps_visual_logits_as_residual():
    graph = build_label_graph([
        {"labels": [2, 4]}, {"labels": [2, 4]}, {"labels": [2]}, {"labels": [4]},
    ], [2, 4], min_support=2)
    model = DefectClassifier(2, pretrained=False, label_graph=graph)
    output = model(torch.zeros(2, 3, 64, 64))
    assert output["logits"].shape == output["raw_logits"].shape == (2, 2)
    assert model.graph_alpha_logit.requires_grad
