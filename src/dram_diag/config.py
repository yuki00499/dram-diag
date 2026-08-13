from pathlib import Path

def load_config(path="configs/default.yaml"):
    try:
        import yaml
    except ImportError:
        return {}
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = Path(__file__).resolve().parents[2] / config_path
    if not config_path.exists():
        return {}
    with config_path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
