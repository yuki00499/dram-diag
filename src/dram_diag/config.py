from pathlib import Path

def load_config(path="configs/default.yaml"):
    try:
        import yaml
    except ImportError:
        return {}
    with Path(path).open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
