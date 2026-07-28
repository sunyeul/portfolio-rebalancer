# Qlib Capability 및 1단계 레짐 신호 검증 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** IPS Pilot 운영 경로와 분리된 Qlib capability gate를 만들고, 불변 Toss 시장 데이터로 현재 레짐 신호의 향후 20·60거래일 하방 위험을 재현·평가한다.

**Architecture:** 기본 IPS Pilot 환경에는 Qlib을 추가하지 않고 `research/qlib_validation/`의 별도 uv 환경에 `pyqlib==0.9.7`을 고정한다. 읽기 전용 SQLite exporter가 활성 정책과 네 벤치마크·현재 정책 종목 데이터를 불변 스냅샷으로 만들며, Qlib adapter 적합성과 결정론적 Stage 1 replay는 서로 독립적으로 실행된다. Qlib factor, 다중시장 시각 또는 재현성 gate가 실패해도 Stage 1은 기존 `evaluate_dynamic_allocation()`으로 실행되지만 Qlib은 부적합으로 기록된다.

**Tech Stack:** Python 3.12, uv, pyqlib 0.9.7, pandas(Qlib 의존성), SQLite URI read-only mode, 표준 라이브러리 `argparse`·`dataclasses`·`hashlib`·`statistics`·`zoneinfo`, pytest, ruff

---

## 범위

이번 계획은 다음 두 결과만 만든다.

1. Qlib environment/data-adapter capability 판정
2. 4개 레짐 벤치마크와 현재 정책 종목에 대한 Stage 1 신호 검증 보고서

전체 포트폴리오 NAV, USD/KRW 환산, 거래비용, 정기매수 또는 목표비중 리밸런싱을 계산하지 않는다. `target_policy_verdict`는 항상 `inconclusive`와 `stage2_not_run` 사유로 남긴다. Stage 2는 총수익 가격·환율·캘린더·승인된 비용 데이터가 실제로 준비된 뒤 별도 구현 계획으로 작성한다.

## 파일 구조

| 파일 | 책임 |
|---|---|
| `research/__init__.py` | 연구 패키지 경계 |
| `research/qlib_validation/__init__.py` | Qlib 검증 패키지 경계 |
| `research/qlib_validation/pyproject.toml` | 격리된 Qlib 의존성 |
| `research/qlib_validation/uv.lock` | 정확한 연구 환경 lock |
| `research/qlib_validation/protocol.json` | Stage 1 사전등록 값 |
| `research/qlib_validation/environment.py` | Python/Qlib smoke test |
| `research/qlib_validation/contracts.py` | immutable 입력·replay·metric 타입 |
| `research/qlib_validation/source.py` | SQLite read-only exporter와 UTC 가용 시각 |
| `research/qlib_validation/artifacts.py` | 정책·입력·소스 manifest와 원자적 파일 기록 |
| `research/qlib_validation/capability.py` | Qlib StaticDataLoader parity 및 fail-closed 판정 |
| `research/qlib_validation/replay.py` | 월말 point-in-time 레짐 재현 |
| `research/qlib_validation/metrics.py` | 미래 위험 지표, block bootstrap, 신호 판정 |
| `research/qlib_validation/report.py` | Stage 1 orchestration 및 summary 생성 |
| `research/qlib_validation/cli.py` | JSON stdout 연구 CLI |
| `research/qlib_validation/README.md` | 공식 문서 링크와 프로젝트 실행 진입점 |
| `tests/research/conftest.py` | 공유 immutable snapshot fixture |
| `tests/research/` | 연구 전용 단위·통합 테스트 |
| `.gitignore` | 실행 산출물 제외 |

기존 `pyproject.toml`, 운영 CLI, API, SQLite schema와 평가 저장소는 수정하지 않는다.

### Task 1: 격리 환경과 사전등록 프로토콜

**Files:**
- Create: `research/__init__.py`
- Create: `research/qlib_validation/__init__.py`
- Create: `research/qlib_validation/pyproject.toml`
- Create: `research/qlib_validation/protocol.json`
- Create: `research/qlib_validation/environment.py`
- Create: `tests/research/test_qlib_environment.py`
- Modify: `.gitignore`
- Generate: `research/qlib_validation/uv.lock`

- [ ] **Step 1: 패키지 경계와 격리 환경 정의**

`research/__init__.py`와 `research/qlib_validation/__init__.py`는 빈 파일로 만든다. `research/qlib_validation/pyproject.toml`을 다음 내용으로 만든다.

```toml
[project]
name = "ips-pilot-qlib-validation"
version = "0.1.0"
requires-python = ">=3.12,<3.13"
dependencies = [
    "pyqlib==0.9.7",
]

[dependency-groups]
dev = [
    "pytest>=8.0.0",
    "ruff>=0.14.2",
]

[tool.uv]
package = false
```

`research/qlib_validation/protocol.json`을 다음처럼 고정한다.

```json
{
  "protocol_version": "stage1-v1",
  "minimum_history": 200,
  "maximum_staleness_days": 7,
  "horizons": [20, 60],
  "minimum_risk_off_episodes": 3,
  "bootstrap": {
    "block_months": 3,
    "samples": 10000,
    "seed": 20260728,
    "confidence": 0.95
  },
  "availability_rules": {
    "KR": {"timezone": "Asia/Seoul", "conservative_close": "16:00"},
    "US": {"timezone": "America/New_York", "conservative_close": "16:00"}
  },
  "signal_rule": {
    "primary_metric": "max_drawdown",
    "comparison": "risk_off_minus_other",
    "required_baskets": ["benchmarks", "policy_instruments"],
    "supported_upper_ci_below": 0.0,
    "not_supported_point_at_or_above": 0.0
  }
}
```

- [ ] **Step 2: 환경 계약의 실패 테스트 작성**

```python
# tests/research/test_qlib_environment.py
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
```

- [ ] **Step 3: 연구 환경을 lock하고 실패 확인**

Run:

```bash
rtk uv lock --project research/qlib_validation
rtk uv run --project research/qlib_validation pytest tests/research/test_qlib_environment.py -q
```

Expected: FAIL with `ModuleNotFoundError: research.qlib_validation.environment`.

- [ ] **Step 4: 최소 smoke-test 구현**

```python
# research/qlib_validation/environment.py
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
```

`.gitignore`에 다음 한 줄을 추가한다.

```gitignore
research/qlib_validation/artifacts/
```

- [ ] **Step 5: 격리 환경과 기존 동작 확인**

Run:

```bash
rtk uv run --project research/qlib_validation pytest tests/research/test_qlib_environment.py -q
rtk uv run --project research/qlib_validation python -m research.qlib_validation.environment
rtk uv run pytest tests/test_dynamic_allocation.py -q
```

Expected: 연구 테스트 PASS, stdout JSON의 `pyqlib`이 `0.9.7`, 기존 동적 배분 테스트 PASS.

- [ ] **Step 6: 환경 계약 커밋**

```bash
rtk git add .gitignore research/__init__.py research/qlib_validation/__init__.py research/qlib_validation/pyproject.toml research/qlib_validation/uv.lock research/qlib_validation/protocol.json research/qlib_validation/environment.py tests/research/test_qlib_environment.py
rtk git commit -m "build: isolate qlib validation environment"
```

### Task 2: Read-only Toss 입력 스냅샷

**Files:**
- Create: `research/qlib_validation/contracts.py`
- Create: `research/qlib_validation/source.py`
- Create: `tests/research/test_qlib_source.py`

- [ ] **Step 1: read-only·가용 시각 실패 테스트 작성**

```python
# tests/research/test_qlib_source.py
from datetime import UTC, datetime
import sqlite3

import pytest

from services.dynamic_allocation import allocation_benchmarks
from storage.database import initialize_database
from storage.market_store import insert_candles
from storage.policy_store import get_active_policy
from research.qlib_validation.source import SourceError, _available_at, load_snapshot, open_readonly


def test_snapshot_reads_active_policy_and_never_opens_database_for_write(tmp_path, monkeypatch):
    database = tmp_path / "portfolio.sqlite3"
    monkeypatch.setenv("PORTFOLIO_DB_PATH", str(database))
    initialize_database()
    active = get_active_policy()
    assert active is not None

    for spec in allocation_benchmarks(active["policy"]):
        insert_candles([
            {
                "source_kind": spec["source_kind"],
                "market_country": spec["market_country"],
                "symbol": spec["symbol"],
                "interval": "1d",
                "candle_at": "2026-01-30T09:00:00+09:00" if spec["market_country"] == "KR" else "2026-01-30T09:30:00-05:00",
                "currency": "KRW" if spec["market_country"] == "KR" else "USD",
                "open_price": 100.0,
                "high_price": 101.0,
                "low_price": 99.0,
                "close_price": 100.5,
                "volume": 1000.0,
                "adjusted": spec["source_kind"] == "stock",
                "adjusted_supported": spec["source_kind"] == "stock",
            }
        ])

    snapshot = load_snapshot(database, as_of=datetime(2026, 2, 1, tzinfo=UTC))
    assert snapshot.policy_record["policy_hash"] == active["policy_hash"]
    assert {item.key for item in snapshot.benchmark_specs} == {
        "US/SPY", "US/QQQ", "KR/KOSPI", "KR/KOSDAQ"
    }
    assert all(candle.factor is None for candle in snapshot.candles)

    with open_readonly(database) as conn:
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            conn.execute("CREATE TABLE forbidden_write (id INTEGER)")


def test_availability_rules_fail_closed_for_naive_time_and_unknown_market():
    with pytest.raises(SourceError, match="timezone-aware"):
        _available_at(datetime(2026, 1, 30, 9, 0), "KR")
    with pytest.raises(SourceError, match="unsupported market"):
        _available_at(datetime(2026, 1, 30, 9, 0, tzinfo=UTC), "JP")
```

- [ ] **Step 2: 실패 확인**

Run:

```bash
rtk uv run pytest tests/research/test_qlib_source.py -q
```

Expected: FAIL because `contracts.py` and `source.py` do not exist.

- [ ] **Step 3: immutable 계약 타입 구현**

```python
# research/qlib_validation/contracts.py
from dataclasses import asdict, dataclass
from datetime import date, datetime
from typing import Any, Literal


@dataclass(frozen=True)
class SeriesSpec:
    key: str
    source_kind: str
    market_country: str
    symbol: str
    weight: float
    role: Literal["benchmark", "policy_instrument"]


@dataclass(frozen=True)
class Candle:
    key: str
    source_kind: str
    market_country: str
    symbol: str
    session_date: date
    candle_at: datetime
    available_at: datetime
    currency: str
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume: float
    adjusted: bool
    adjusted_supported: bool
    factor: float | None

    def evaluator_row(self) -> dict[str, Any]:
        return {
            "candle_at": self.candle_at.isoformat(),
            "close_price": self.close_price,
        }

    def record(self) -> dict[str, Any]:
        value = asdict(self)
        value["session_date"] = self.session_date.isoformat()
        value["candle_at"] = self.candle_at.isoformat()
        value["available_at"] = self.available_at.isoformat()
        return value


@dataclass(frozen=True)
class SourceSnapshot:
    policy_record: dict[str, Any]
    benchmark_specs: tuple[SeriesSpec, ...]
    policy_specs: tuple[SeriesSpec, ...]
    candles: tuple[Candle, ...]

    def candles_for(self, key: str) -> tuple[Candle, ...]:
        return tuple(item for item in self.candles if item.key == key)
```

- [ ] **Step 4: SQLite URI read-only exporter 구현**

`source.py`는 기존 상태 함수를 호출해 DB를 생성하지 않고 `mode=ro` URI만 사용한다. `factor`는 현재 schema에 없으므로 반드시 `None`으로 둔다.

```python
# research/qlib_validation/source.py
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, date, datetime, time
from functools import cache
import json
from pathlib import Path
import sqlite3
from zoneinfo import ZoneInfo

from services.dynamic_allocation import allocation_benchmarks
from research.qlib_validation.contracts import Candle, SeriesSpec, SourceSnapshot


PROTOCOL_PATH = Path(__file__).with_name("protocol.json")


@cache
def _availability() -> dict[str, tuple[ZoneInfo, time]]:
    rules = json.loads(PROTOCOL_PATH.read_text())["availability_rules"]
    return {
        market: (
            ZoneInfo(value["timezone"]),
            time.fromisoformat(value["conservative_close"]),
        )
        for market, value in rules.items()
    }


class SourceError(RuntimeError):
    pass


@contextmanager
def open_readonly(path: Path) -> Iterator[sqlite3.Connection]:
    if not path.is_file():
        raise SourceError(f"database not found: {path}")
    conn = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
    try:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA query_only = ON")
        yield conn
    finally:
        conn.close()


def _available_at(candle_at: datetime, market_country: str) -> tuple[datetime, date]:
    if candle_at.tzinfo is None:
        raise SourceError("candle_at must be timezone-aware")
    availability = _availability()
    if market_country not in availability:
        raise SourceError(f"unsupported market availability rule: {market_country}")
    zone, conservative_close = availability[market_country]
    local = candle_at.astimezone(zone)
    available = datetime.combine(local.date(), conservative_close, zone).astimezone(UTC)
    return available, local.date()


def _active_policy(conn: sqlite3.Connection) -> dict[str, object]:
    row = conn.execute(
        """
        SELECT id, account_alias, version, policy_json, policy_hash, created_at
        FROM ips_policy_versions
        WHERE account_alias = 'toss-brokerage' AND superseded_at IS NULL
        ORDER BY version DESC, id DESC
        LIMIT 1
        """
    ).fetchone()
    if row is None:
        raise SourceError("active policy not found")
    return {
        "id": int(row["id"]),
        "account_alias": row["account_alias"],
        "version": int(row["version"]),
        "policy": json.loads(row["policy_json"]),
        "policy_hash": row["policy_hash"],
        "created_at": row["created_at"],
    }


def _specs(policy: dict[str, object]) -> tuple[tuple[SeriesSpec, ...], tuple[SeriesSpec, ...]]:
    benchmarks = tuple(
        SeriesSpec(
            key=str(item["key"]),
            source_kind=str(item["source_kind"]),
            market_country=str(item["market_country"]),
            symbol=str(item["symbol"]),
            weight=float(item["weight"]),
            role="benchmark",
        )
        for item in allocation_benchmarks(policy)
    )
    instruments = tuple(policy.get("instruments", []))
    total = sum(float(item["target"]) for item in instruments)
    if not instruments or total <= 0:
        raise SourceError("policy instruments are required")
    policy_specs = tuple(
        SeriesSpec(
            key=f"{item['market_country']}/{item['symbol']}",
            source_kind="stock",
            market_country=str(item["market_country"]),
            symbol=str(item["symbol"]),
            weight=float(item["target"]) / total,
            role="policy_instrument",
        )
        for item in instruments
    )
    return benchmarks, policy_specs


def _load_series(conn: sqlite3.Connection, spec: SeriesSpec, as_of: datetime) -> list[Candle]:
    rows = conn.execute(
        """
        SELECT source_kind, market_country, symbol, candle_at, currency,
               open_price, high_price, low_price, close_price, volume,
               adjusted, adjusted_supported
        FROM toss_market_candles
        WHERE source_kind = ? AND market_country = ? AND symbol = ? AND interval = '1d'
          AND (? != 'stock' OR (adjusted = 1 AND adjusted_supported = 1))
        ORDER BY datetime(candle_at), id
        """,
        (spec.source_kind, spec.market_country, spec.symbol, spec.source_kind),
    ).fetchall()
    candles: list[Candle] = []
    for row in rows:
        candle_at = datetime.fromisoformat(row["candle_at"])
        available_at, session_date = _available_at(candle_at, spec.market_country)
        if available_at > as_of:
            continue
        candles.append(Candle(
            key=spec.key,
            source_kind=spec.source_kind,
            market_country=spec.market_country,
            symbol=spec.symbol,
            session_date=session_date,
            candle_at=candle_at,
            available_at=available_at,
            currency=str(row["currency"]),
            open_price=float(row["open_price"]),
            high_price=float(row["high_price"]),
            low_price=float(row["low_price"]),
            close_price=float(row["close_price"]),
            volume=float(row["volume"]),
            adjusted=bool(row["adjusted"]),
            adjusted_supported=bool(row["adjusted_supported"]),
            factor=None,
        ))
    return candles


def load_snapshot(path: Path, *, as_of: datetime) -> SourceSnapshot:
    if as_of.tzinfo is None:
        raise SourceError("as_of must be timezone-aware")
    with open_readonly(path) as conn:
        policy_record = _active_policy(conn)
        policy = policy_record["policy"]
        benchmarks, policy_specs = _specs(policy)
        unique = {item.key: item for item in (*benchmarks, *policy_specs)}
        candles = tuple(
            candle
            for key in sorted(unique)
            for candle in _load_series(conn, unique[key], as_of)
        )
    return SourceSnapshot(policy_record, benchmarks, policy_specs, candles)
```

- [ ] **Step 5: exporter 테스트 실행**

Run:

```bash
rtk uv run pytest tests/research/test_qlib_source.py -q
rtk uv run pytest tests/test_market_store.py tests/test_policy_store.py -q
```

Expected: 모두 PASS. 테스트 DB 외의 파일은 생성·수정되지 않는다.

- [ ] **Step 6: read-only exporter 커밋**

```bash
rtk git add research/qlib_validation/contracts.py research/qlib_validation/source.py tests/research/test_qlib_source.py
rtk git commit -m "feat: export immutable qlib research inputs"
```

### Task 3: 재현 가능한 입력·소스 manifest

**Files:**
- Create: `research/qlib_validation/artifacts.py`
- Create: `tests/research/test_qlib_artifacts.py`

- [ ] **Step 1: 결정론·덮어쓰기 방지 실패 테스트 작성**

```python
# tests/research/test_qlib_artifacts.py
import json
from pathlib import Path

import pytest

from research.qlib_validation.artifacts import ArtifactError, canonical_bytes, write_inputs


def test_write_inputs_is_canonical_hashed_and_never_overwrites(snapshot_factory, tmp_path):
    snapshot = snapshot_factory()
    repository_root = Path(__file__).resolve().parents[2]
    result = write_inputs(snapshot, tmp_path / "run", repository_root=repository_root)
    policy = (tmp_path / "run" / "policy.json").read_bytes()
    manifest = json.loads((tmp_path / "run" / "input-manifest.json").read_text())

    assert policy == canonical_bytes(snapshot.policy_record["policy"])
    assert manifest["policy_hash"] == snapshot.policy_record["policy_hash"]
    assert manifest["files"]["candles.jsonl"]["sha256"] == result["candles_sha256"]
    assert manifest["series"][snapshot.benchmark_specs[0].key]["rows"] == 1
    source = json.loads((tmp_path / "run" / "source-manifest.json").read_text())
    assert "research/qlib_validation/protocol.json" in source["files"]
    assert "research/qlib_validation/uv.lock" in source["files"]

    with pytest.raises(ArtifactError, match="already exists"):
        write_inputs(snapshot, tmp_path / "run", repository_root=tmp_path)
```

`tests/research/conftest.py`에 다음 공유 fixture를 추가한다.

```python
from copy import deepcopy
from datetime import UTC, datetime, timedelta

import pytest

from services.dynamic_allocation import allocation_benchmarks
from storage.policy_store import DEFAULT_POLICY, policy_hash
from research.qlib_validation.contracts import Candle, SeriesSpec, SourceSnapshot


@pytest.fixture
def snapshot_factory():
    def build(days: int = 1) -> SourceSnapshot:
        policy = deepcopy(DEFAULT_POLICY)
        benchmark_specs = tuple(
            SeriesSpec(
                key=item["key"],
                source_kind=item["source_kind"],
                market_country=item["market_country"],
                symbol=item["symbol"],
                weight=float(item["weight"]),
                role="benchmark",
            )
            for item in allocation_benchmarks(policy)
        )
        instruments = tuple(policy["instruments"])
        invested_total = sum(float(item["target"]) for item in instruments)
        policy_specs = tuple(
            SeriesSpec(
                key=f"{item['market_country']}/{item['symbol']}",
                source_kind="stock",
                market_country=item["market_country"],
                symbol=item["symbol"],
                weight=float(item["target"]) / invested_total,
                role="policy_instrument",
            )
            for item in instruments
        )
        unique = {item.key: item for item in (*benchmark_specs, *policy_specs)}
        sessions = []
        cursor = datetime(2023, 1, 2, 21, 0, tzinfo=UTC)
        while len(sessions) < days:
            if cursor.weekday() < 5:
                sessions.append(cursor)
            cursor += timedelta(days=1)
        candles = tuple(
            Candle(
                key=spec.key,
                source_kind=spec.source_kind,
                market_country=spec.market_country,
                symbol=spec.symbol,
                session_date=point.date(),
                candle_at=point,
                available_at=point,
                currency="KRW" if spec.market_country == "KR" else "USD",
                open_price=100.0 + index,
                high_price=101.0 + index,
                low_price=99.0 + index,
                close_price=100.5 + index,
                volume=1000.0 + index,
                adjusted=spec.source_kind == "stock",
                adjusted_supported=spec.source_kind == "stock",
                factor=None,
            )
            for spec in unique.values()
            for index, point in enumerate(sessions)
        )
        return SourceSnapshot(
            policy_record={
                "id": 1,
                "account_alias": "toss-brokerage",
                "version": 1,
                "policy": policy,
                "policy_hash": policy_hash(policy),
                "created_at": "2023-01-01T00:00:00+00:00",
            },
            benchmark_specs=benchmark_specs,
            policy_specs=policy_specs,
            candles=candles,
        )

    return build


@pytest.fixture
def long_snapshot(snapshot_factory):
    return snapshot_factory(days=340)
```

- [ ] **Step 2: 실패 확인**

Run:

```bash
rtk uv run pytest tests/research/test_qlib_artifacts.py -q
```

Expected: FAIL because `artifacts.py` does not exist.

- [ ] **Step 3: canonical writer와 source manifest 구현**

```python
# research/qlib_validation/artifacts.py
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any

from research.qlib_validation.contracts import SourceSnapshot


RELEVANT_SOURCE_PATHS = (
    "services/dynamic_allocation.py",
    "research/qlib_validation",
)

FIXED_REPRODUCIBILITY_FILES = (
    "services/dynamic_allocation.py",
    "research/qlib_validation/protocol.json",
    "research/qlib_validation/pyproject.toml",
    "research/qlib_validation/uv.lock",
)


class ArtifactError(RuntimeError):
    pass


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _atomic_write(path: Path, payload: bytes) -> None:
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        handle.write(payload)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def write_json(path: Path, value: Any) -> None:
    _atomic_write(path, canonical_bytes(value))


def _digest(payload: bytes) -> str:
    return sha256(payload).hexdigest()


def _source_manifest(repository_root: Path) -> dict[str, Any]:
    files = set((repository_root / "research/qlib_validation").rglob("*.py"))
    files.update(repository_root / value for value in FIXED_REPRODUCIBILITY_FILES)
    hashes = {
        str(path.relative_to(repository_root)): sha256(path.read_bytes()).hexdigest()
        for path in sorted(files)
        if path.is_file()
    }
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repository_root, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    dirty = bool(subprocess.run(
        ["git", "status", "--porcelain", "--", *RELEVANT_SOURCE_PATHS],
        cwd=repository_root, check=True, capture_output=True, text=True,
    ).stdout.strip())
    return {"git_commit": commit, "relevant_source_dirty": dirty, "files": hashes}


def write_inputs(snapshot: SourceSnapshot, run_dir: Path, *, repository_root: Path) -> dict[str, Any]:
    if run_dir.exists():
        raise ArtifactError(f"run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    policy_payload = canonical_bytes(snapshot.policy_record["policy"])
    candle_lines = b"".join(canonical_bytes(item.record()) for item in snapshot.candles)
    _atomic_write(run_dir / "policy.json", policy_payload)
    _atomic_write(run_dir / "candles.jsonl", candle_lines)
    source = _source_manifest(repository_root)
    _atomic_write(run_dir / "source-manifest.json", canonical_bytes(source))
    by_key = {
        spec.key: sorted(snapshot.candles_for(spec.key), key=lambda item: item.available_at)
        for spec in (*snapshot.benchmark_specs, *snapshot.policy_specs)
    }
    manifest = {
        "policy_hash": snapshot.policy_record["policy_hash"],
        "series": {
            key: {
                "rows": len(items),
                "min_available_at": items[0].available_at.isoformat() if items else None,
                "max_available_at": items[-1].available_at.isoformat() if items else None,
            }
            for key, items in sorted(by_key.items())
        },
        "files": {
            "policy.json": {"sha256": _digest(policy_payload)},
            "candles.jsonl": {"sha256": _digest(candle_lines), "rows": len(snapshot.candles)},
        },
    }
    _atomic_write(run_dir / "input-manifest.json", canonical_bytes(manifest))
    return {
        "policy_sha256": manifest["files"]["policy.json"]["sha256"],
        "candles_sha256": manifest["files"]["candles.jsonl"]["sha256"],
        "relevant_source_dirty": source["relevant_source_dirty"],
        "source_manifest": source,
        "input_manifest": manifest,
    }
```

- [ ] **Step 4: manifest 테스트 실행**

Run:

```bash
rtk uv run pytest tests/research/test_qlib_artifacts.py -q
```

Expected: PASS, 동일 fixture의 SHA-256이 반복 실행에서 동일하다.

- [ ] **Step 5: manifest 커밋**

```bash
rtk git add research/qlib_validation/artifacts.py tests/research/conftest.py tests/research/test_qlib_artifacts.py
rtk git commit -m "feat: record reproducible qlib research manifests"
```

### Task 4: Qlib capability gate

**Files:**
- Create: `research/qlib_validation/capability.py`
- Create: `tests/research/test_qlib_capability.py`

- [ ] **Step 1: factor 누락과 Qlib round-trip 테스트 작성**

```python
# tests/research/test_qlib_capability.py
from dataclasses import replace
from importlib.util import find_spec

import pytest

from research.qlib_validation.capability import assess_capability, static_roundtrip
from research.qlib_validation.contracts import SourceSnapshot


pytestmark = pytest.mark.skipif(find_spec("qlib") is None, reason="Qlib research environment only")


def test_real_toss_shape_fails_closed_when_factor_is_unavailable(snapshot_factory):
    result = assess_capability(snapshot_factory())
    assert result["data_adapter_suitable"] is False
    assert "factor_unavailable" in result["reasons"]
    assert result["backtest_engine_suitable"] is False


def test_static_loader_round_trip_preserves_verified_fixture(snapshot_factory):
    snapshot = snapshot_factory()
    candles = tuple(replace(item, factor=1.0) for item in snapshot.candles)
    verified = SourceSnapshot(
        snapshot.policy_record, snapshot.benchmark_specs, snapshot.policy_specs, candles
    )
    assert static_roundtrip(verified) == {"rows": len(candles), "matched": True}
```

- [ ] **Step 2: 실패 확인**

Run:

```bash
rtk uv run --project research/qlib_validation pytest tests/research/test_qlib_capability.py -q
```

Expected: FAIL because `capability.py` does not exist.

- [ ] **Step 3: lazy Qlib adapter와 fail-closed 판정 구현**

```python
# research/qlib_validation/capability.py
from importlib import metadata
from typing import Any

from research.qlib_validation.contracts import SourceSnapshot


def static_roundtrip(snapshot: SourceSnapshot) -> dict[str, Any]:
    import pandas as pd
    from pandas.testing import assert_frame_equal
    from qlib.data.dataset.loader import StaticDataLoader

    records = [
        {
            "datetime": item.session_date.isoformat(),
            "instrument": item.key.replace("/", "_"),
            "open": item.open_price,
            "high": item.high_price,
            "low": item.low_price,
            "close": item.close_price,
            "volume": item.volume,
            "factor": item.factor,
        }
        for item in snapshot.candles
    ]
    frame = pd.DataFrame.from_records(records)
    frame["datetime"] = pd.to_datetime(frame["datetime"])
    expected = frame.set_index(["datetime", "instrument"]).sort_index()
    loaded = StaticDataLoader(expected).load().sort_index()
    assert_frame_equal(loaded, expected, check_dtype=False, check_names=True)
    return {"rows": len(expected), "matched": True}


def assess_capability(snapshot: SourceSnapshot) -> dict[str, Any]:
    reasons: list[str] = []
    if not snapshot.candles or any(item.factor is None for item in snapshot.candles):
        reasons.append("factor_unavailable")
    if {item.market_country for item in snapshot.benchmark_specs} != {"KR", "US"}:
        reasons.append("required_markets_missing")
    if any(item.available_at.tzinfo is None for item in snapshot.candles):
        reasons.append("utc_availability_missing")
    matched = False
    if not reasons:
        matched = static_roundtrip(snapshot)["matched"]
        if not matched:
            reasons.append("static_loader_mismatch")
    return {
        "pyqlib": metadata.version("pyqlib"),
        "data_adapter_suitable": not reasons and matched,
        "backtest_engine_suitable": False,
        "backtest_reason": "stage2_not_evaluated",
        "reasons": reasons,
    }
```

- [ ] **Step 4: capability와 기본 환경 회귀 확인**

Run:

```bash
rtk uv run --project research/qlib_validation pytest tests/research/test_qlib_capability.py -q
rtk uv run pytest tests/test_dynamic_allocation.py tests/test_market_store.py -q
```

Expected: Qlib fixture round-trip PASS, 실제 Toss shape fixture는 `factor_unavailable`로 예상된 fail-closed 결과, 기존 테스트 PASS.

- [ ] **Step 5: capability gate 커밋**

```bash
rtk git add research/qlib_validation/capability.py tests/research/test_qlib_capability.py
rtk git commit -m "feat: gate qlib data adapter capability"
```

### Task 5: 월말 point-in-time 레짐 replay

**Files:**
- Modify: `research/qlib_validation/contracts.py`
- Create: `research/qlib_validation/replay.py`
- Create: `tests/research/test_qlib_replay.py`

- [ ] **Step 1: 미래 데이터 차단과 기존 evaluator 위임 테스트 작성**

```python
# tests/research/test_qlib_replay.py
from research.qlib_validation.replay import replay_regimes


def test_replay_uses_each_markets_last_same_month_candle_and_no_future_rows(long_snapshot):
    seen = []

    def evaluator(series_by_key, *, active_policy, last_change_at, now):
        seen.append((series_by_key, now))
        return {"regime": "risk_off", "reason": "regime_target_change"}

    points = replay_regimes(long_snapshot, evaluator=evaluator)
    assert points
    assert all(point.regime == "risk_off" for point in points)
    for series_map, decision_timestamp in seen:
        for rows in series_map.values():
            assert len(rows) >= 200
            assert max(row["available_at"] for row in rows) <= decision_timestamp.isoformat()


def test_replay_output_does_not_copy_ips_status(long_snapshot):
    points = replay_regimes(long_snapshot)
    assert all("status" not in point.record() for point in points)
```

- [ ] **Step 2: ReplayPoint 계약 추가 후 실패 확인**

`contracts.py`에 다음 타입을 추가한다.

```python
@dataclass(frozen=True)
class ReplayPoint:
    month: str
    decision_timestamp: datetime
    regime: str | None
    reason: str
    cutoffs: dict[str, str]

    def record(self) -> dict[str, Any]:
        return {
            "month": self.month,
            "decision_timestamp": self.decision_timestamp.isoformat(),
            "regime": self.regime,
            "reason": self.reason,
            "cutoffs": dict(sorted(self.cutoffs.items())),
        }
```

Run:

```bash
rtk uv run pytest tests/research/test_qlib_replay.py -q
```

Expected: FAIL because `replay.py` does not exist.

- [ ] **Step 3: Stage 1 replay 구현**

Stage 1은 레짐 신호 자체만 검증하므로 정책 후보를 활성화하거나 `last_change_at`을 합성하지 않는다. 30일 cooldown과 목표 정책 상태 전이는 Stage 2 계획의 책임이다.

```python
# research/qlib_validation/replay.py
from collections.abc import Callable
from datetime import datetime
from typing import Any

from services.dynamic_allocation import build_neutral_policy, evaluate_dynamic_allocation
from research.qlib_validation.contracts import Candle, ReplayPoint, SourceSnapshot


Evaluator = Callable[..., dict[str, Any]]


def _by_month(candles: tuple[Candle, ...]) -> dict[str, list[Candle]]:
    grouped: dict[str, list[Candle]] = {}
    for candle in candles:
        grouped.setdefault(candle.session_date.strftime("%Y-%m"), []).append(candle)
    return grouped


def replay_regimes(
    snapshot: SourceSnapshot,
    *,
    evaluator: Evaluator = evaluate_dynamic_allocation,
    minimum_history: int = 200,
) -> list[ReplayPoint]:
    policy = snapshot.policy_record["policy"]
    neutral = build_neutral_policy(policy)
    all_specs = {
        spec.key: spec
        for spec in (*snapshot.benchmark_specs, *snapshot.policy_specs)
    }
    monthly = {
        spec.key: _by_month(snapshot.candles_for(spec.key))
        for spec in all_specs.values()
    }
    common_months = sorted(set.intersection(*(set(value) for value in monthly.values())))
    points: list[ReplayPoint] = []
    for month in common_months:
        selected = {
            spec.key: max(monthly[spec.key][month], key=lambda item: item.session_date)
            for spec in snapshot.benchmark_specs
        }
        decision_timestamp = max(item.available_at for item in selected.values())
        histories = {
            key: [
                item
                for item in snapshot.candles_for(key)
                if item.available_at <= decision_timestamp
            ]
            for key in all_specs
        }
        if any(len(items) < minimum_history for items in histories.values()):
            continue
        series_by_key: dict[str, list[dict[str, Any]]] = {}
        cutoffs: dict[str, str] = {}
        for key, cutoff in selected.items():
            cutoffs[key] = cutoff.session_date.isoformat()
            series_by_key[key] = [
                {**item.evaluator_row(), "available_at": item.available_at.isoformat()}
                for item in sorted(histories[key], key=lambda value: value.available_at)
                if item.session_date <= cutoff.session_date
            ]
        result = evaluator(
            series_by_key,
            active_policy=neutral,
            last_change_at=None,
            now=decision_timestamp,
        )
        points.append(ReplayPoint(
            month=month,
            decision_timestamp=decision_timestamp,
            regime=result.get("regime"),
            reason=str(result.get("reason", "unknown")),
            cutoffs=cutoffs,
        ))
    return points
```

- [ ] **Step 4: evaluator parity와 replay 테스트 실행**

Run:

```bash
rtk uv run pytest tests/research/test_qlib_replay.py tests/test_dynamic_allocation.py -q
```

Expected: PASS. `ReplayPoint`에는 `OK`, `Watch`, `Review`, `Action`이나 `status` 필드가 없다.

- [ ] **Step 5: replay 커밋**

```bash
rtk git add research/qlib_validation/contracts.py research/qlib_validation/replay.py tests/research/test_qlib_replay.py
rtk git commit -m "feat: replay allocation regimes point in time"
```

### Task 6: 미래 위험 지표와 block-bootstrap 판정

**Files:**
- Modify: `research/qlib_validation/contracts.py`
- Create: `research/qlib_validation/metrics.py`
- Create: `tests/research/test_qlib_metrics.py`

여기서 `basket` 지표는 각 종목의 20·60세션 하방 지표를 현재 정책 가중치로 평균한 **복합 신호 지표**다. 서로 다른 거래일·통화의 가격을 합쳐 포트폴리오 NAV를 재구성한 값이 아니며, NAV·환율·비용을 요구하는 해석은 Stage 2로 미룬다.

- [ ] **Step 1: 위험 지표와 판정 경계 테스트 작성**

```python
# tests/research/test_qlib_metrics.py
import pytest

from research.qlib_validation.contracts import SourceSnapshot
from research.qlib_validation.metrics import (
    block_bootstrap_effect,
    build_forward_observations,
    downside_metrics,
    signal_verdict,
)
from research.qlib_validation.replay import replay_regimes


def test_downside_metrics_measure_drawdown_volatility_worst_day_and_recovery():
    result = downside_metrics([100.0, 90.0, 80.0, 90.0, 100.0])
    assert result["max_drawdown"] == pytest.approx(-0.2)
    assert result["worst_daily_return"] == pytest.approx(-1.0 / 9.0)
    assert result["recovery_sessions"] == 2
    assert result["annualized_volatility"] > 0


def test_block_bootstrap_is_seeded_and_reproducible():
    observations = [
        ("risk_off" if index % 3 == 0 else "neutral", -0.12 if index % 3 == 0 else -0.03)
        for index in range(18)
    ]
    first = block_bootstrap_effect(observations, samples=200, seed=20260728)
    second = block_bootstrap_effect(observations, samples=200, seed=20260728)
    assert first == second
    assert first["estimate"] < 0.0


def test_signal_supported_requires_three_episodes_and_negative_upper_ci():
    effects = {
        ("benchmarks", 20): {"estimate": -0.05, "ci_low": -0.08, "ci_high": -0.01},
        ("benchmarks", 60): {"estimate": -0.07, "ci_low": -0.10, "ci_high": -0.02},
        ("policy_instruments", 20): {"estimate": -0.04, "ci_low": -0.07, "ci_high": -0.01},
        ("policy_instruments", 60): {"estimate": -0.06, "ci_low": -0.09, "ci_high": -0.02},
    }
    assert signal_verdict(effects, risk_off_episodes=3, complete_coverage=True)["verdict"] == "supported"
    assert signal_verdict(effects, risk_off_episodes=2, complete_coverage=True)["verdict"] == "inconclusive"


def test_non_negative_effect_is_not_supported_when_data_is_complete():
    effects = {
        ("benchmarks", 20): {"estimate": 0.01, "ci_low": -0.02, "ci_high": 0.03},
        ("benchmarks", 60): {"estimate": -0.02, "ci_low": -0.05, "ci_high": 0.01},
        ("policy_instruments", 20): {"estimate": -0.01, "ci_low": -0.04, "ci_high": 0.02},
        ("policy_instruments", 60): {"estimate": -0.03, "ci_low": -0.06, "ci_high": 0.01},
    }
    assert signal_verdict(effects, risk_off_episodes=3, complete_coverage=True)["verdict"] == "not_supported"
    assert signal_verdict(
        effects,
        risk_off_episodes=3,
        complete_coverage=True,
        source_fresh=False,
    )["reason"] == "source_stale"
    assert signal_verdict(
        effects,
        risk_off_episodes=3,
        complete_coverage=True,
        replay_complete=False,
    )["reason"] == "replay_incomplete"
```

- [ ] **Step 2: 실패 확인**

Run:

```bash
rtk uv run pytest tests/research/test_qlib_metrics.py -q
```

Expected: FAIL because `metrics.py` does not exist.

- [ ] **Step 3: 하방 지표와 사전등록 판정 구현**

```python
# research/qlib_validation/metrics.py
from math import sqrt
import random
from statistics import mean, pstdev
from typing import Any

from research.qlib_validation.contracts import ReplayPoint, SourceSnapshot


REQUIRED_EFFECTS = {
    ("benchmarks", 20),
    ("benchmarks", 60),
    ("policy_instruments", 20),
    ("policy_instruments", 60),
}


def downside_metrics(closes: list[float]) -> dict[str, float | int | None]:
    returns = [current / previous - 1.0 for previous, current in zip(closes, closes[1:])]
    peak = closes[0]
    max_drawdown = 0.0
    trough_index = 0
    peak_before_trough = closes[0]
    for index, close in enumerate(closes):
        if close > peak:
            peak = close
        drawdown = close / peak - 1.0
        if drawdown < max_drawdown:
            max_drawdown = drawdown
            trough_index = index
            peak_before_trough = peak
    recovery = next(
        (index - trough_index for index in range(trough_index + 1, len(closes)) if closes[index] >= peak_before_trough),
        None,
    )
    return {
        "annualized_volatility": pstdev(returns) * sqrt(252) if len(returns) > 1 else 0.0,
        "max_drawdown": max_drawdown,
        "worst_daily_return": min(returns) if returns else 0.0,
        "recovery_sessions": recovery,
    }


def block_bootstrap_effect(
    observations: list[tuple[str, float]],
    *,
    block_months: int = 3,
    samples: int = 10000,
    seed: int = 20260728,
    confidence: float = 0.95,
) -> dict[str, float]:
    def difference(items: list[tuple[str, float]]) -> float | None:
        risk_off = [value for regime, value in items if regime == "risk_off"]
        other = [value for regime, value in items if regime in {"neutral", "risk_on"}]
        return mean(risk_off) - mean(other) if risk_off and other else None

    estimate = difference(observations)
    if estimate is None:
        raise ValueError("both risk_off and comparison observations are required")
    rng = random.Random(seed)
    generated: list[float] = []
    attempts = 0
    while len(generated) < samples and attempts < samples * 20:
        attempts += 1
        sample: list[tuple[str, float]] = []
        while len(sample) < len(observations):
            start = rng.randrange(len(observations))
            sample.extend(observations[(start + offset) % len(observations)] for offset in range(block_months))
        value = difference(sample[:len(observations)])
        if value is not None:
            generated.append(value)
    if len(generated) < samples:
        raise ValueError("bootstrap could not preserve both comparison groups")
    generated.sort()
    tail = (1.0 - confidence) / 2.0
    return {
        "estimate": estimate,
        "ci_low": generated[int(samples * tail)],
        "ci_high": generated[min(samples - 1, int(samples * (1.0 - tail)))],
    }


def _next_closes(
    snapshot: SourceSnapshot,
    key: str,
    decision_timestamp,
    horizon: int,
) -> list[float] | None:
    series = sorted(snapshot.candles_for(key), key=lambda item: item.available_at)
    eligible = [index for index, item in enumerate(series) if item.available_at <= decision_timestamp]
    if not eligible:
        return None
    start = eligible[-1]
    window = series[start:start + horizon + 1]
    return [item.close_price for item in window] if len(window) == horizon + 1 else None


def _risk_off_episodes(points: list[ReplayPoint]) -> int:
    count = 0
    previous = None
    for point in points:
        if point.regime == "risk_off" and previous != "risk_off":
            count += 1
        previous = point.regime
    return count


def build_forward_observations(
    snapshot: SourceSnapshot,
    replay_points: list[ReplayPoint],
    horizons: tuple[int, ...] = (20, 60),
    *,
    block_months: int = 3,
    samples: int = 10000,
    seed: int = 20260728,
    confidence: float = 0.95,
) -> dict[str, Any]:
    series_rows: list[dict[str, Any]] = []
    basket_rows: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    baskets = {
        "benchmarks": snapshot.benchmark_specs,
        "policy_instruments": snapshot.policy_specs,
    }
    for point in replay_points:
        if point.regime not in {"risk_on", "neutral", "risk_off"}:
            continue
        for horizon in horizons:
            reference_missing = [
                spec.key
                for spec in snapshot.benchmark_specs
                if _next_closes(snapshot, spec.key, point.decision_timestamp, horizon) is None
            ]
            if reference_missing:
                right_censored = len(reference_missing) == len(snapshot.benchmark_specs)
                missing.extend(
                    {
                        "month": point.month,
                        "key": key,
                        "horizon": horizon,
                        "reason": "right_censored" if right_censored else "benchmark_forward_history_short",
                        "blocking": not right_censored,
                    }
                    for key in reference_missing
                )
                continue
            for basket, specs in baskets.items():
                measured: list[tuple[float, dict[str, float | int | None]]] = []
                for spec in specs:
                    closes = _next_closes(snapshot, spec.key, point.decision_timestamp, horizon)
                    if closes is None:
                        missing.append({
                            "month": point.month,
                            "key": spec.key,
                            "horizon": horizon,
                            "reason": "forward_history_short",
                            "blocking": basket == "policy_instruments",
                        })
                        continue
                    values = downside_metrics(closes)
                    measured.append((spec.weight, values))
                    series_rows.append({
                        "scope": "series",
                        "month": point.month,
                        "regime": point.regime,
                        "basket": basket,
                        "key": spec.key,
                        "horizon": horizon,
                        **values,
                    })
                if len(measured) != len(specs):
                    continue
                weight_total = sum(weight for weight, _ in measured)
                basket_rows.append({
                    "scope": "basket",
                    "month": point.month,
                    "regime": point.regime,
                    "basket": basket,
                    "horizon": horizon,
                    "max_drawdown": sum(weight * float(value["max_drawdown"]) for weight, value in measured) / weight_total,
                    "annualized_volatility": sum(weight * float(value["annualized_volatility"]) for weight, value in measured) / weight_total,
                    "worst_daily_return": sum(weight * float(value["worst_daily_return"]) for weight, value in measured) / weight_total,
                })
    effects: dict[tuple[str, int], dict[str, float]] = {}
    for basket, horizon in REQUIRED_EFFECTS:
        observations = [
            (str(row["regime"]), float(row["max_drawdown"]))
            for row in basket_rows
            if row["basket"] == basket and row["horizon"] == horizon
        ]
        try:
            effects[(basket, horizon)] = block_bootstrap_effect(
                observations,
                block_months=block_months,
                samples=samples,
                seed=seed,
                confidence=confidence,
            )
        except ValueError:
            continue
    serializable = {
        f"{basket}:{horizon}": value
        for (basket, horizon), value in sorted(effects.items())
    }
    return {
        "rows": [*series_rows, *basket_rows],
        "missing": missing,
        "analysis": {
            "effects": effects,
            "effects_serializable": serializable,
            "risk_off_episodes": _risk_off_episodes(replay_points),
        },
    }


def signal_verdict(
    effects: dict[tuple[str, int], dict[str, float]],
    *,
    risk_off_episodes: int,
    complete_coverage: bool,
    reproducible: bool = True,
    source_fresh: bool = True,
    replay_complete: bool = True,
    minimum_risk_off_episodes: int = 3,
) -> dict[str, Any]:
    if not complete_coverage:
        return {"verdict": "inconclusive", "reason": "policy_instrument_coverage_incomplete"}
    if not reproducible:
        return {"verdict": "inconclusive", "reason": "relevant_source_dirty"}
    if not source_fresh:
        return {"verdict": "inconclusive", "reason": "source_stale"}
    if not replay_complete:
        return {"verdict": "inconclusive", "reason": "replay_incomplete"}
    if risk_off_episodes < minimum_risk_off_episodes:
        return {"verdict": "inconclusive", "reason": "risk_off_episodes_below_three"}
    if set(effects) != REQUIRED_EFFECTS:
        return {"verdict": "inconclusive", "reason": "required_effects_missing"}
    if any(item["estimate"] >= 0.0 for item in effects.values()):
        return {"verdict": "not_supported", "reason": "max_drawdown_direction_failed"}
    if all(item["ci_high"] < 0.0 for item in effects.values()):
        return {"verdict": "supported", "reason": "downside_signal_confirmed"}
    return {"verdict": "inconclusive", "reason": "confidence_interval_crosses_zero"}
```

- [ ] **Step 4: 우측 절단과 정책 종목 누락 테스트 추가**

`tests/research/test_qlib_metrics.py`에 다음 검사를 추가한다.

```python
def test_latest_month_is_right_censored_without_blocking(long_snapshot):
    point = replay_regimes(long_snapshot)[-1]
    result = build_forward_observations(long_snapshot, [point], (60,))
    assert result["missing"]
    assert all(item["reason"] == "right_censored" for item in result["missing"])
    assert all(item["blocking"] is False for item in result["missing"])


def test_missing_policy_instrument_blocks_signal_verdict(long_snapshot):
    points = replay_regimes(long_snapshot)
    point = points[len(points) // 2]
    benchmark_keys = {item.key for item in long_snapshot.benchmark_specs}
    missing_spec = next(item for item in long_snapshot.policy_specs if item.key not in benchmark_keys)
    shortened = tuple(
        item
        for item in long_snapshot.candles
        if item.key != missing_spec.key or item.available_at <= point.decision_timestamp
    )
    snapshot = SourceSnapshot(
        long_snapshot.policy_record,
        long_snapshot.benchmark_specs,
        long_snapshot.policy_specs,
        shortened,
    )
    result = build_forward_observations(snapshot, [point], (20,))
    blocking = [item for item in result["missing"] if item["blocking"]]
    assert blocking == [{
        "month": point.month,
        "key": missing_spec.key,
        "horizon": 20,
        "reason": "forward_history_short",
        "blocking": True,
    }]


def test_partial_benchmark_truncation_is_not_treated_as_right_censoring(long_snapshot):
    points = replay_regimes(long_snapshot)
    point = points[len(points) // 2]
    missing_spec = long_snapshot.benchmark_specs[0]
    shortened = tuple(
        item
        for item in long_snapshot.candles
        if item.key != missing_spec.key or item.available_at <= point.decision_timestamp
    )
    snapshot = SourceSnapshot(
        long_snapshot.policy_record,
        long_snapshot.benchmark_specs,
        long_snapshot.policy_specs,
        shortened,
    )
    result = build_forward_observations(snapshot, [point], (20,))
    assert any(
        item["key"] == missing_spec.key
        and item["reason"] == "benchmark_forward_history_short"
        and item["blocking"] is True
        for item in result["missing"]
    )
```

- [ ] **Step 5: metrics 테스트 실행**

Run:

```bash
rtk uv run pytest tests/research/test_qlib_metrics.py -q
```

Expected: PASS. 10,000회 테스트가 느리면 테스트 입력에서만 `samples=200`을 전달하고 production 기본값은 변경하지 않는다.

- [ ] **Step 6: metrics 커밋**

```bash
rtk git add research/qlib_validation/contracts.py research/qlib_validation/metrics.py tests/research/test_qlib_metrics.py
rtk git commit -m "feat: evaluate forward downside regime evidence"
```

### Task 7: Stage 1 보고서와 JSON CLI

**Files:**
- Create: `research/qlib_validation/report.py`
- Create: `research/qlib_validation/cli.py`
- Create: `research/qlib_validation/README.md`
- Create: `tests/research/test_qlib_report.py`
- Create: `tests/research/test_qlib_cli.py`
- Create: `tests/research/test_qlib_integration.py`

- [ ] **Step 1: report 안전 계약 실패 테스트 작성**

```python
# tests/research/test_qlib_report.py
from datetime import UTC, datetime

from research.qlib_validation.report import run_stage1


def test_stage1_summary_never_claims_stage2_or_emits_execution_fields(
    monkeypatch, snapshot_factory, tmp_path
):
    snapshot = snapshot_factory(days=80)

    def fake_write_inputs(value, run_dir, *, repository_root):
        assert value is snapshot
        run_dir.mkdir(parents=True)
        return {
            "relevant_source_dirty": False,
            "source_manifest": {"git_commit": "test", "files": {}},
            "input_manifest": {"policy_hash": "test", "files": {}},
        }

    monkeypatch.setattr("research.qlib_validation.report.load_snapshot", lambda *args, **kwargs: snapshot)
    monkeypatch.setattr("research.qlib_validation.report.write_inputs", fake_write_inputs)
    monkeypatch.setattr(
        "research.qlib_validation.report.assess_capability",
        lambda value: {
            "data_adapter_suitable": False,
            "backtest_engine_suitable": False,
            "reasons": ["factor_unavailable"],
        },
    )
    monkeypatch.setattr(
        "research.qlib_validation.report.environment_info",
        lambda: {"python": "3.12.test", "pyqlib": "0.9.7"},
    )
    monkeypatch.setattr(
        "research.qlib_validation.report.replay_regimes",
        lambda value, **kwargs: [],
    )
    monkeypatch.setattr(
        "research.qlib_validation.report.build_forward_observations",
        lambda *args, **kwargs: {
            "rows": [],
            "missing": [],
            "analysis": {"effects": {}, "effects_serializable": {}, "risk_off_episodes": 0},
        },
    )
    summary = run_stage1(
        database=tmp_path / "ignored.sqlite3",
        as_of=datetime(2026, 7, 28, tzinfo=UTC),
        output=tmp_path / "artifacts",
    )
    assert summary["regime_signal_verdict"] in {"supported", "inconclusive", "not_supported"}
    assert summary["target_policy_verdict"] == "inconclusive"
    assert summary["target_policy_reason"] == "stage2_not_run"
    assert (tmp_path / "artifacts" / summary["run_id"] / "manifest.json").is_file()

    forbidden = {"buy", "sell", "execute", "order_size", "status"}

    def keys(value):
        if isinstance(value, dict):
            return set(value) | set().union(*(keys(item) for item in value.values()))
        if isinstance(value, list):
            return set().union(*(keys(item) for item in value), set())
        return set()

    assert forbidden.isdisjoint(keys(summary))
```

```python
# tests/research/test_qlib_cli.py
import json

from research.qlib_validation.cli import main


def test_cli_writes_one_json_object_to_stdout(monkeypatch, capsys, tmp_path):
    def noisy_run(**kwargs):
        print("dependency progress noise")
        return {
            "run_id": "fixed-run",
            "regime_signal_verdict": "inconclusive",
            "target_policy_verdict": "inconclusive",
        }

    monkeypatch.setattr(
        "research.qlib_validation.cli.run_stage1",
        noisy_run,
    )
    code = main([
        "stage1", "--db", str(tmp_path / "db.sqlite3"),
        "--as-of", "2026-07-28T00:00:00+00:00",
        "--output", str(tmp_path / "artifacts"),
    ])
    assert code == 0
    assert json.loads(capsys.readouterr().out)["run_id"] == "fixed-run"
```

```python
# tests/research/test_qlib_integration.py
from hashlib import sha256
from importlib.util import find_spec
import json

import pytest

from research.qlib_validation.cli import main
from services.dynamic_allocation import allocation_benchmarks
from storage.database import initialize_database
from storage.market_store import insert_candles
from storage.policy_store import get_active_policy


pytestmark = pytest.mark.skipif(
    find_spec("qlib") is None,
    reason="Qlib research environment only",
)


def test_real_cli_keeps_fixture_database_bytes_unchanged(monkeypatch, capsys, tmp_path):
    database = tmp_path / "fixture.sqlite3"
    monkeypatch.setenv("PORTFOLIO_DB_PATH", str(database))
    initialize_database()
    active = get_active_policy()
    assert active is not None
    policy_specs = [
        {
            "source_kind": "stock",
            "market_country": item["market_country"],
            "symbol": item["symbol"],
        }
        for item in active["policy"]["instruments"]
    ]
    unique_specs = {
        (item["source_kind"], item["market_country"], item["symbol"]): item
        for item in [*allocation_benchmarks(active["policy"]), *policy_specs]
    }
    for spec in unique_specs.values():
        is_stock = spec["source_kind"] == "stock"
        insert_candles([{
            "source_kind": spec["source_kind"],
            "market_country": spec["market_country"],
            "symbol": spec["symbol"],
            "interval": "1d",
            "candle_at": (
                "2026-01-30T09:00:00+09:00"
                if spec["market_country"] == "KR"
                else "2026-01-30T09:30:00-05:00"
            ),
            "currency": "KRW" if spec["market_country"] == "KR" else "USD",
            "open_price": 100.0,
            "high_price": 101.0,
            "low_price": 99.0,
            "close_price": 100.5,
            "volume": 1000.0,
            "adjusted": is_stock,
            "adjusted_supported": is_stock,
        }])
    before = sha256(database.read_bytes()).hexdigest()
    code = main([
        "stage1",
        "--db", str(database),
        "--as-of", "2026-02-01T00:00:00+00:00",
        "--output", str(tmp_path / "artifacts"),
    ])
    result = json.loads(capsys.readouterr().out)
    assert code == 0
    assert result["regime_signal_verdict"] == "inconclusive"
    assert result["target_policy_verdict"] == "inconclusive"
    assert sha256(database.read_bytes()).hexdigest() == before
```

- [ ] **Step 2: 실패 확인**

Run:

```bash
rtk uv run pytest tests/research/test_qlib_report.py tests/research/test_qlib_cli.py tests/research/test_qlib_integration.py -q
```

Expected: FAIL because `report.py` and `cli.py` do not exist.

- [ ] **Step 3: report orchestration 구현**

```python
# research/qlib_validation/report.py
from datetime import UTC, datetime, timedelta
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from research.qlib_validation.artifacts import canonical_bytes, write_inputs, write_json
from research.qlib_validation.capability import assess_capability
from research.qlib_validation.contracts import SourceSnapshot
from research.qlib_validation.environment import environment_info
from research.qlib_validation.metrics import build_forward_observations, signal_verdict
from research.qlib_validation.replay import replay_regimes
from research.qlib_validation.source import load_snapshot


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = Path(__file__).with_name("protocol.json")


def _protocol() -> tuple[dict[str, Any], str]:
    value = json.loads(PROTOCOL_PATH.read_text())
    return value, sha256(canonical_bytes(value)).hexdigest()


def _stale_series(
    snapshot: SourceSnapshot,
    *,
    as_of: datetime,
    maximum_staleness_days: int,
) -> list[str]:
    cutoff = as_of - timedelta(days=maximum_staleness_days)
    specs = {
        spec.key: spec
        for spec in (*snapshot.benchmark_specs, *snapshot.policy_specs)
    }
    return sorted(
        key
        for key in specs
        if not snapshot.candles_for(key)
        or max(item.available_at for item in snapshot.candles_for(key)) < cutoff
    )


def run_stage1(*, database: Path, as_of: datetime, output: Path) -> dict[str, Any]:
    if as_of.tzinfo is None:
        raise ValueError("as_of must be timezone-aware")
    as_of = as_of.astimezone(UTC)
    protocol, protocol_hash = _protocol()
    snapshot = load_snapshot(database, as_of=as_of)
    input_fingerprint = sha256(
        canonical_bytes({
            "as_of": as_of.isoformat(),
            "policy_hash": snapshot.policy_record["policy_hash"],
            "protocol_hash": protocol_hash,
            "candles": [item.record() for item in snapshot.candles],
        })
    ).hexdigest()
    run_id = f"{as_of.strftime('%Y%m%dT%H%M%SZ')}-{input_fingerprint[:12]}"
    run_dir = output / run_id
    input_manifest = write_inputs(snapshot, run_dir, repository_root=ROOT)
    capability = assess_capability(snapshot)
    points = replay_regimes(
        snapshot,
        minimum_history=int(protocol["minimum_history"]),
    )
    forward = build_forward_observations(
        snapshot,
        points,
        tuple(protocol["horizons"]),
        **protocol["bootstrap"],
    )
    analysis = forward["analysis"]
    stale_series = _stale_series(
        snapshot,
        as_of=as_of,
        maximum_staleness_days=int(protocol["maximum_staleness_days"]),
    )
    unclassified_months = [
        point.month
        for point in points
        if point.regime not in {"risk_on", "neutral", "risk_off"}
    ]
    verdict = signal_verdict(
        analysis["effects"],
        risk_off_episodes=analysis["risk_off_episodes"],
        complete_coverage=not any(item["blocking"] for item in forward["missing"]),
        reproducible=not input_manifest["relevant_source_dirty"],
        source_fresh=not stale_series,
        replay_complete=not unclassified_months,
        minimum_risk_off_episodes=int(protocol["minimum_risk_off_episodes"]),
    )
    manifest = {
        "run_id": run_id,
        "as_of": as_of.isoformat(),
        "protocol_hash": protocol_hash,
        "policy_hash": snapshot.policy_record["policy_hash"],
        "environment": environment_info(),
        "qlib_capability": capability,
        "source_manifest": input_manifest["source_manifest"],
        "input_manifest": input_manifest["input_manifest"],
    }
    summary = {
        "run_id": run_id,
        "as_of": as_of.isoformat(),
        "protocol_hash": protocol_hash,
        "policy_hash": snapshot.policy_record["policy_hash"],
        "qlib_capability": capability,
        "regime_signal_verdict": verdict["verdict"],
        "regime_signal_reason": verdict["reason"],
        "target_policy_verdict": "inconclusive",
        "target_policy_reason": "stage2_not_run",
        "coverage_missing": forward["missing"],
        "stale_series": stale_series,
        "unclassified_months": unclassified_months,
        "effects": analysis["effects_serializable"],
    }
    write_json(run_dir / "manifest.json", manifest)
    write_json(run_dir / "replay.json", [item.record() for item in points])
    write_json(run_dir / "stage1-metrics.json", forward["rows"])
    write_json(run_dir / "summary.json", summary)
    return summary
```

`build_forward_observations()`는 Task 6에서 `rows`, `missing`, `analysis.effects`, `analysis.effects_serializable`, `analysis.risk_off_episodes`를 모두 반환하도록 완성한다. `effects`는 tuple key를 사용하고 `effects_serializable`은 `benchmarks:20` 같은 정렬된 문자열 key를 사용한다.

- [ ] **Step 4: argparse JSON CLI 구현**

```python
# research/qlib_validation/cli.py
import argparse
from contextlib import redirect_stdout
from datetime import datetime
from io import StringIO
import json
from pathlib import Path
from typing import Sequence

from research.qlib_validation.report import run_stage1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="qlib-validation")
    commands = parser.add_subparsers(dest="command", required=True)
    stage1 = commands.add_parser("stage1")
    stage1.add_argument("--db", type=Path, required=True)
    stage1.add_argument("--as-of", required=True)
    stage1.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        with redirect_stdout(StringIO()):
            result = run_stage1(
                database=args.db,
                as_of=datetime.fromisoformat(args.as_of),
                output=args.output,
            )
    except Exception as exc:
        print(json.dumps({"ok": False, "error": type(exc).__name__, "message": str(exc)}, sort_keys=True))
        return 1
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: 최소 실행 문서 작성**

`research/qlib_validation/README.md`에는 설치 튜토리얼을 복제하지 않고 다음 내용만 둔다.

````markdown
# Qlib validation research

Qlib 설치와 API 설명은 [공식 stable 문서](https://qlib.readthedocs.io/en/stable/)를 기준으로 한다.
이 디렉터리는 `pyqlib==0.9.7` lock, IPS Pilot 전용 입력 계약과 재현 가능한 실행만 소유한다.

환경 확인:

```bash
rtk uv run --project research/qlib_validation python -m research.qlib_validation.environment
```

Stage 1 실행:

```bash
rtk uv run --project research/qlib_validation python -m research.qlib_validation.cli stage1 --db data/portfolio_rebalancer.sqlite3 --as-of 2026-07-28T00:00:00+00:00 --output research/qlib_validation/artifacts
```

출력은 연구 근거이며 IPS 상태, 정책 활성화 또는 주문 권한이 아니다. Qlib factor가 원천 데이터에 없으면 capability는 `factor_unavailable`로 닫히며 값을 만들어내지 않는다.
````

- [ ] **Step 6: report·CLI 테스트 실행**

Run:

```bash
rtk uv run --project research/qlib_validation pytest tests/research -q
rtk uv run pytest tests/research -q
```

Expected: 연구 환경에서는 전부 PASS. 기본 환경에서는 `pytest.mark.skipif`가 붙은 Qlib 전용 테스트만 SKIP되고 나머지는 PASS.

- [ ] **Step 7: report·CLI 커밋**

```bash
rtk git add research/qlib_validation/report.py research/qlib_validation/cli.py research/qlib_validation/README.md tests/research/test_qlib_report.py tests/research/test_qlib_cli.py tests/research/test_qlib_integration.py
rtk git commit -m "feat: report qlib regime signal validation"
```

### Task 8: 전체 검증과 가드레일 감사

**Files:**
- Modify only files that fail the checks below

- [ ] **Step 1: placeholder와 금지 필드 정적 검색**

Run:

```bash
rtk rg -n "NotImplementedError|pass[[:space:]]+#|buy|sell|execute|order_size|\"status\"" research/qlib_validation tests/research
```

Expected: 미완성 구현 표식, 주문 필드, 연구 출력의 `status`가 없다. 테스트가 금지 문자열 부재를 검증하기 위해 포함한 문자열만 검색되면 해당 테스트 위치를 확인하고 유지한다.

- [ ] **Step 2: format과 lint**

Run:

```bash
rtk uv run ruff format --check research/qlib_validation tests/research
rtk uv run ruff check research/qlib_validation tests/research
```

Expected: 두 명령 모두 PASS. 실패하면 `rtk uv run ruff format research/qlib_validation tests/research`로 기계적 포맷 후 다시 검사한다.

- [ ] **Step 3: 연구·운영 회귀 테스트**

Run:

```bash
rtk uv run --project research/qlib_validation pytest tests/research -q
rtk uv run pytest -q
```

Expected: 연구 환경 전체 PASS, 기본 환경 전체 PASS와 Qlib 전용 테스트의 예상 SKIP. 라이브 Toss 호출은 발생하지 않는다.

- [ ] **Step 4: 실제 CLI read-only smoke 실행**

Task 7의 통합 테스트가 `tmp_path`에 SQLite fixture를 만들고 실제 CLI를 실행한 뒤 DB 파일 SHA-256이 동일한지 검증한다. 원본 `data/` DB는 열지 않는다.

```bash
rtk uv run --project research/qlib_validation pytest tests/research/test_qlib_integration.py -q
```

Expected: PASS. 실제 CLI 결과는 두 verdict 모두 `inconclusive`이고 SQLite fixture의 파일 해시는 실행 전후 동일하다.

- [ ] **Step 5: 최종 변경 범위 확인과 커밋**

Run:

```bash
rtk git status --short
rtk git diff --check
```

Expected: 이 계획의 research·tests·`.gitignore` 변경만 커밋 대상이다. 기존 사용자 변경은 stage하지 않는다.

```bash
rtk git add research/qlib_validation tests/research .gitignore
rtk git commit -m "test: verify qlib validation guardrails"
```

검사로 인한 수정이 없으면 이 마지막 커밋은 생략한다. 수정이 생겼을 때만 관련 파일을 명시적으로 stage하고 기존 사용자 변경은 제외한다.

## 자체 점검 매핑

| 승인된 설계 요구사항 | 구현·검증 위치 |
|---|---|
| 공개 설치법을 복제하지 않고 정확한 버전과 공식 링크만 소유 | Task 1 `pyproject.toml`·`uv.lock`, Task 7 README |
| 운영 의존성과 Qlib 연구 환경 분리 | Task 1 환경 테스트, Task 8 기본 환경 회귀 테스트 |
| normalized Toss 데이터만 사용하고 DB·정책 상태를 쓰지 않음 | Task 2 `mode=ro`·`query_only`, Task 8 원본 해시 smoke |
| 원천에 없는 factor를 합성하지 않고 Qlib 적합성을 실패 폐쇄 | Task 2 `factor=None`, Task 4 `factor_unavailable` gate |
| KR·US의 데이터 가용 시각과 UTC cutoff로 미래 참조 방지 | Task 1 protocol, Task 2 가용 시각 테스트, Task 5 replay 테스트 |
| 네 벤치마크와 모든 현재 정책 종목의 최대 공통 적격 기간 사용 | Task 5 공통 월·종목별 200세션 gate, Task 6 coverage 테스트 |
| 오래됐거나 일부 벤치마크만 끊긴 입력을 우측 절단으로 오인하지 않음 | Task 1 7일 freshness, Task 6 부분 truncation 테스트, Task 7 stale gate |
| protocol·환경·정책·입력·관련 소스의 재현 가능성 기록 | Task 3 source/input manifest, Task 7 run manifest |
| 3개월 block, 10,000회, seed·95% CI와 최소 3개 에피소드 적용 | Task 1 protocol, Task 6 bootstrap·판정, Task 7 protocol 전달 |
| Stage 1 신호와 Stage 2 목표 정책 결론을 분리 | Task 7 `regime_signal_verdict`, 고정 `target_policy_verdict=inconclusive` |
| IPS 상태·주문 지시·운영 경로 변경 금지 | Task 7 출력 계약, Task 8 정적 검색과 전체 회귀 테스트 |

계획의 모든 생성 파일에는 실패 테스트, 최소 구현, 관련 회귀 검증과 의도별 커밋 경계가 있다. Stage 2에 필요한 NAV·FX·총수익·비용 데이터나 정책 상태 전이는 이 계획 어디에서도 암묵적으로 대체하지 않는다.

## 완료 조건

- 기본 `uv sync`와 IPS Pilot 테스트는 Qlib 없이 동작한다.
- 연구 lock은 `pyqlib==0.9.7`과 Python 3.12 계약을 재현한다.
- exporter는 존재하는 SQLite를 `mode=ro`와 `PRAGMA query_only=ON`으로만 연다.
- 정책 원문, candle JSONL, protocol, source와 input SHA-256이 실행 산출물에 남는다.
- 현재 schema에 없는 Qlib factor를 합성하지 않고 capability를 `factor_unavailable`로 닫는다.
- Stage 1 evaluator 입력에 각 월말 cutoff 이후 candle이 포함되지 않는다.
- 종목별 최신 candle이 as-of보다 7일 넘게 오래됐거나 일부 벤치마크만 미래 구간이 끊기면 신호 판정은 `inconclusive`다.
- 정책 종목 하나라도 미래 20·60세션 coverage가 부족하면 신호 판정은 `inconclusive`다.
- 독립 `risk_off` 에피소드 3개, 두 basket·두 horizon의 음의 효과와 95% 신뢰구간 조건을 모두 통과해야만 `supported`다.
- 결과는 `regime_signal_verdict`와 `target_policy_verdict`를 사용하며 IPS 상태나 주문 필드를 만들지 않는다.
- Stage 2 계산, 운영 API·CLI 연결, SQLite 쓰기와 정책 활성화는 존재하지 않는다.
