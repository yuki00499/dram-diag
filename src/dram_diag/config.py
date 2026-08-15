from pathlib import Path


def load_yaml(path):
    import yaml

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"找不到配置文件: {config_path}")
    with config_path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}
