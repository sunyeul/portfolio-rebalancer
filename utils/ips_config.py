"""IPS 설정 로딩 유틸리티."""

from pathlib import Path

import yaml


DEFAULT_IPS_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "ips.yaml"


def load_ips_config(path: str | Path = DEFAULT_IPS_CONFIG_PATH) -> dict:
    """IPS YAML 설정 파일을 dict로 로드합니다."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
