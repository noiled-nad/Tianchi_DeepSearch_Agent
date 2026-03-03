from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict

import yaml


_PROMPT_DIR = Path(__file__).resolve().parent / "prompts"


@lru_cache(maxsize=64)
def load_prompt(file_name: str, key: str) -> str:
    """Load a prompt string from deepresearch/prompts/*.yaml with in-process cache."""
    path = _PROMPT_DIR / file_name
    if not path.exists():
        raise FileNotFoundError(f"prompt file not found: {path}")

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, Dict) or key not in data:
        raise KeyError(f"prompt key '{key}' not found in {path}")

    val = data[key]
    if not isinstance(val, str):
        raise TypeError(f"prompt '{key}' in {path} must be string")
    return val
