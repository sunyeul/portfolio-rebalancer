from importlib import import_module, metadata
import json
import platform


def environment_info() -> dict[str, object]:
    import_module("qlib")
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "pyqlib": metadata.version("pyqlib"),
        "pandas": metadata.version("pandas"),
        "qlib_imported": True,
    }


if __name__ == "__main__":
    print(json.dumps(environment_info(), sort_keys=True))
