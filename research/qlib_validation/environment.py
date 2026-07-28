import json
import platform
from importlib import import_module, metadata


def environment_info() -> dict[str, object]:
    import_module("qlib")
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "pyqlib": metadata.version("pyqlib"),
        "pandas": metadata.version("pandas"),
        "qlib_imported": True,
    }


def main() -> None:
    try:
        print(json.dumps(environment_info(), sort_keys=True))
    except Exception as error:
        payload = {
            "error": f"{type(error).__name__}: {error}"[:240],
            "qlib_imported": False,
        }
        print(json.dumps(payload, sort_keys=True))
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
