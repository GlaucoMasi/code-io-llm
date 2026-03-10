import yaml
from types import SimpleNamespace
from pathlib import Path

_CONFIG_DIR = Path(__file__).parent

def _load(name: str) -> SimpleNamespace:
    with open(_CONFIG_DIR / f"{name}.yaml", "r") as f:
        data = yaml.safe_load(f)
    return SimpleNamespace(**data)

small = _load("small")
