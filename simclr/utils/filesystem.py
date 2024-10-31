from typing import Any, Dict
import yaml


def load_yaml(path: str, encoding: str = "utf-8") -> Dict[Any, Any]:
    with open(path, encoding=encoding) as file:
        content = yaml.safe_load(file)
    return content
