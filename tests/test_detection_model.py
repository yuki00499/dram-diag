import pytest

torch = pytest.importorskip("torch")

from dram_diag.detection import (GlobalDiagnosisHead, P2DetectionHead, ReviewPolicy,
                                 consistency_weight, hierarchical_losses,
                                 post_nms_presence, review_decision,
                                 soft_presence_pool)


def test_hierarchical_heads_and_p2_shapes():
    head = GlobalDiagnosisHead([32, 64, 128], defect_classes=4, quality_classes=3)
    output = head([
        torch.randn(2, 32, 80, 80), torch.randn(2, 64, 40, 40), torch.randn(2, 128, 20, 20)])
    assert output["global_defect_logits"].shape == (2, 4)
    assert output["quality_logits"].shape == (2, 3)
    assert output["usability_logits"].shape == (2, 3)
    p2 = P2DetectionHead(32, 4)
    prediction = p2(torch.randn(2, 32, 160, 160))
    assert prediction["class_logits"].shape == (2, 4, 160, 160)
    assert prediction["box_distribution"].shape == (2, 64, 160, 160)


def test_hierarchical_loss_masks_unusable_detection_target_and_warms_up():
    outputs = {
        "global_defect_logits": torch.zeros(2, 2, requires_grad=True),
        "quality_logits": torch.zeros(2, 1, requires_grad=True),
        "usability_logits": torch.zeros(2, 3, requires_grad=True),
    }
    targets = {
        "defect_presence": torch.tensor([[1, 0], [0, 1]]),
        "quality_attributes": torch.tensor([[0], [1]]),
        "usability": torch.tensor([0, 2]),
        "diagnostic_mask": torch.tensor([1, 0]),
    }
    result = hierarchical_losses(outputs, targets, torch.randn(2, 10, 2), 5, {
        "consistency_weight": .2, "consistency_warmup_epochs": 10})
    assert result["hierarchical_total"].requires_grad
    assert result["consistency_weight"].item() == pytest.approx(.1)
    assert consistency_weight(10, 10, .2) == pytest.approx(.2)
    assert soft_presence_pool(torch.zeros(2, 4, 2)).shape == (2, 2)


def test_review_policy_explains_global_local_disagreement():
    detections = [{"class_id": 0, "confidence": .6}, {"class_id": 0, "confidence": .5}]
    presence = post_nms_presence(detections, [0, 1])
    assert presence[0] == pytest.approx(.8)
    result = review_decision({0: .1, 1: .9}, presence, {"blur": .8},
                             {"usable": .2, "review": .8, "unusable": 0}, ReviewPolicy())
    assert result["review_required"]
    codes = {item["code"] for item in result["review_reasons"]}
    assert {"box_without_global", "global_without_box", "quality_review", "quality_degradation"} <= codes


def test_yolo26_p2_and_joint_loss_integration(tmp_path, monkeypatch):
    monkeypatch.setenv("YOLO_CONFIG_DIR", str(tmp_path))
    pytest.importorskip("ultralytics")
    from copy import copy
    from ultralytics.utils import DEFAULT_CFG
    from dram_diag.detection_training import build_hierarchical_types

    model_type, _ = build_hierarchical_types()
    model = model_type(
        "yolo26n-p2.yaml", nc=2, ch=3, quality_classes=1,
        hierarchy_config={"consistency_weight": .2}, verbose=False)
    assert model.stride.tolist() == [4.0, 8.0, 16.0, 32.0]
    model.args = copy(DEFAULT_CFG)
    model.train()
    batch = {
        "img": torch.rand(2, 3, 64, 64),
        "batch_idx": torch.tensor([0]),
        "cls": torch.tensor([[0.]]),
        "bboxes": torch.tensor([[.5, .5, .25, .25]]),
        "defect_presence": torch.tensor([[1., 0.], [0., 0.]]),
        "quality_attributes": torch.zeros(2, 1),
        "usability": torch.tensor([0, 0]),
        "diagnostic_mask": torch.ones(2),
        "global_img": torch.rand(2, 3, 64, 64),
        "global_defect_presence": torch.tensor([[1., 0.], [0., 0.]]),
        "global_quality_attributes": torch.zeros(2, 1),
        "global_usability": torch.tensor([0, 2]),
        "global_diagnostic_mask": torch.tensor([1., 0.]),
    }
    loss, items = model(batch)
    assert loss.shape == (4,)
    assert torch.isfinite(loss).all()
    assert "hierarchical_loss" in items
    loss.sum().backward()
    assert model.global_head.defect_head.weight.grad.abs().sum() > 0
