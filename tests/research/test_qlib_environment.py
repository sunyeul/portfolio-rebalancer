import json
import tomllib
from importlib import metadata
from importlib.util import find_spec
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

from research.qlib_validation import environment

ROOT = Path(__file__).resolve().parents[2]


def test_runtime_dependencies_do_not_include_pyqlib():
    runtime = tomllib.loads((ROOT / "pyproject.toml").read_text())
    dependency_names = {
        canonicalize_name(Requirement(item).name)
        for item in runtime["project"]["dependencies"]
    }
    assert canonicalize_name("pyqlib") not in dependency_names


@pytest.mark.skipif(find_spec("qlib") is None, reason="Qlib research environment only")
def test_research_environment_pins_qlib():
    info = environment.environment_info()
    assert info["python"].startswith("3.12.")
    assert info["pyqlib"] == "0.9.7"
    assert info["pandas"] == metadata.version("pandas")
    assert isinstance(info["platform"], str) and info["platform"]
    assert isinstance(info["pandas"], str) and info["pandas"]
    assert info["qlib_imported"] is True


def test_main_reports_json_when_qlib_import_fails(monkeypatch, capsys):
    def fail_import(_: str) -> None:
        raise ImportError("missing qlib")

    monkeypatch.setattr(environment, "import_module", fail_import)

    with pytest.raises(SystemExit) as error:
        environment.main()

    payload = json.loads(capsys.readouterr().out)
    assert error.value.code == 1
    assert payload["qlib_imported"] is False
    assert payload["error"] == "ImportError: missing qlib"
