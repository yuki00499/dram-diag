import hashlib
from pathlib import Path


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_checkpoint(checkpoint, manifest):
    if checkpoint.get("protocol_version") != manifest["protocol_version"]:
        raise ValueError("checkpoint 与 manifest 的协议版本不一致")
    if checkpoint.get("manifest_fingerprint") != manifest["manifest_fingerprint"]:
        raise ValueError("checkpoint 与 manifest 指纹不一致")
    classes = manifest.get("types") if manifest.get("task") == "multilabel" else manifest.get("classification_classes", [])
    expected = {int(value): index for index, value in enumerate(sorted(classes))}
    actual = {int(key): int(value) for key, value in checkpoint.get("class_to_idx", {}).items()}
    if actual != expected:
        raise ValueError("checkpoint 类别映射与 manifest 不一致")
    return checkpoint

