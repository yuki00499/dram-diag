from scripts.evaluate_detection_test import false_negative_count


def test_false_negative_count_is_class_and_iou_aware():
    truth = [
        {"class_id": 0, "bbox_xyxy": [0, 0, 10, 10]},
        {"class_id": 1, "bbox_xyxy": [20, 20, 30, 30]},
    ]
    predictions = [
        {"class_id": 0, "bbox_xyxy": [0, 0, 10, 10], "confidence": .9},
        {"class_id": 1, "bbox_xyxy": [0, 0, 10, 10], "confidence": .9},
    ]
    assert false_negative_count(truth, predictions) == 1
