import pytest
import torch

from dram_diag.compat import validate_checkpoint
from dram_diag.model import DefectClassifier, batch_prototype_loss


def test_model_has_21_outputs_without_download():
    output = DefectClassifier(21, pretrained=False)(torch.zeros(2, 3, 64, 64))
    assert output["logits"].shape == (2, 21)
    assert output["embedding"].shape == (2, 512)


def test_prototype_loss_is_finite():
    embeddings = torch.nn.functional.normalize(torch.randn(6, 8), dim=1)
    assert torch.isfinite(batch_prototype_loss(embeddings, torch.tensor([0, 0, 1, 1, 2, 2])))


def test_checkpoint_protocol_mismatch_is_rejected():
    manifest = {"protocol_version": "ge20-v2", "manifest_fingerprint": "abc", "classification_classes": [3, 4]}
    checkpoint = {"protocol_version": "old", "manifest_fingerprint": "abc", "class_to_idx": {3: 0, 4: 1}}
    with pytest.raises(ValueError):
        validate_checkpoint(checkpoint, manifest)
