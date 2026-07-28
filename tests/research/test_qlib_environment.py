from importlib.util import find_spec
from pathlib import Path
import tomllib

import pytest

from research.qlib_validation.environment import environment_info


ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.skipif(find_spec("qlib") is None, reason="Qlib research environment only")


def test_research_environment_pins_qlib_without_changing_runtime_dependencies():
    info = environment_info()
    assert info["python"].startswith("3.12.")
    assert info["pyqlib"] == "0.9.7"
    assert info["qlib_imported"] is True

    runtime = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert all("pyqlib" not in item for item in runtime["project"]["dependencies"])
