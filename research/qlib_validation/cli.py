import argparse
import json
from collections.abc import Sequence
from contextlib import redirect_stdout
from datetime import datetime
from io import StringIO
from pathlib import Path

from research.qlib_validation.report import run_stage1


class CliArgumentError(ValueError):
    """Raised when agent-facing CLI arguments are invalid."""


def run_forecast(**kwargs):
    """Load optional forecast dependencies only for the forecast command."""
    from research.qlib_validation.forecast import run_forecast as implementation

    return implementation(**kwargs)


class JsonArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise CliArgumentError(message)


def _error_payload(error: str, message: str) -> str:
    return json.dumps(
        {"ok": False, "error": error, "message": message[:500]},
        ensure_ascii=False,
        sort_keys=True,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = JsonArgumentParser(prog="qlib-validation")
    commands = parser.add_subparsers(dest="command", required=True)
    stage1 = commands.add_parser("stage1")
    stage1.add_argument("--db", type=Path, required=True)
    stage1.add_argument("--as-of", required=True)
    stage1.add_argument("--output", type=Path, required=True)
    stage1.add_argument(
        "--universe",
        choices=("active-policy", "current-holdings"),
        default="active-policy",
    )
    forecast = commands.add_parser("forecast")
    forecast.add_argument("--db", type=Path, required=True)
    forecast.add_argument("--as-of", required=True)
    forecast.add_argument("--output", type=Path, required=True)
    forecast.add_argument(
        "--universe",
        choices=("active-policy", "current-holdings"),
        default="current-holdings",
    )
    try:
        args = parser.parse_args(argv)
    except CliArgumentError as exc:
        print(_error_payload("ArgumentError", str(exc)))
        return 2
    try:
        with redirect_stdout(StringIO()):
            if args.command == "stage1":
                result = run_stage1(
                    database=args.db,
                    as_of=datetime.fromisoformat(args.as_of),
                    output=args.output,
                    universe=args.universe,
                )
            else:
                result = run_forecast(
                    database=args.db,
                    as_of=datetime.fromisoformat(args.as_of),
                    output=args.output,
                    universe=args.universe,
                )
    except Exception as exc:  # noqa: BLE001 -- CLI must serialize dependency errors.
        print(_error_payload(type(exc).__name__, str(exc)))
        return 1
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
