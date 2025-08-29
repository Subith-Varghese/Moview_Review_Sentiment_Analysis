import os
import yaml
from typing import Any, Dict
from joblib import dump, load

def read_config(path: str = "config.yaml") -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# alias for consistency if needed
read_yaml = read_config

def ensure_dir(path: str):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

def save_joblib(obj, path: str):
    ensure_dir(path)
    dump(obj, path)

def load_joblib(path: str):
    return load(path)
