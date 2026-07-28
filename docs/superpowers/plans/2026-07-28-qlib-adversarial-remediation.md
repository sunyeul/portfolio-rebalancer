# Qlib Adversarial Review Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 적대적 리뷰의 P1·P2 finding을 회귀 테스트와 함께 수정하여 Qlib Stage 1 입력·재현성·CLI 계약을 실패 폐쇄한다.

**Architecture:** 기존 연구 모듈 경계를 유지한다. 입력 검증은 `source.py`, 실행 수명주기는 `artifacts.py`와 `report.py`, 판정 설정은 `metrics.py`, 사용자 오류 직렬화는 `cli.py`가 각각 소유한다.

**Tech Stack:** Python 3.12, SQLite, argparse, pathlib, pytest, Ruff, uv, pyqlib 0.9.7

---

### Task 1: SQLite snapshot과 입력 계약

**Files:**
- Modify: `research/qlib_validation/source.py`
- Modify: `tests/research/test_qlib_source.py`

- [x] 정책 조회와 candle 조회 사이의 동시 커밋이 섞이지 않는 실패 테스트를 작성한다.
- [x] 저장 `policy_hash` 불일치, 필수 벤치마크 4개 identity 불일치, 비정상·중복·과도한 gap candle 실패 테스트를 작성한다.
- [x] `BEGIN` read transaction과 source-level validator를 최소 구현한다.
- [x] `rtk uv run pytest tests/research/test_qlib_source.py -q`를 통과시킨다.

### Task 2: CLI JSON 오류 계약

**Files:**
- Modify: `research/qlib_validation/cli.py`
- Modify: `tests/research/test_qlib_cli.py`

- [x] command/필수 인자 누락의 stdout JSON·exit 2 실패 테스트를 작성한다.
- [x] 사용자 정의 parser error를 구현하고 정상·실행 예외 경로와 함께 검증한다.
- [x] `rtk uv run pytest tests/research/test_qlib_cli.py -q`를 통과시킨다.

### Task 3: 재현성과 원자적 run 승격

**Files:**
- Modify: `research/qlib_validation/artifacts.py`
- Modify: `research/qlib_validation/report.py`
- Modify: `tests/research/test_qlib_artifacts.py`
- Modify: `tests/research/test_qlib_report.py`

- [x] `services/policy_validation.py` hash·dirty 감지 실패 테스트를 작성한다.
- [x] 중간 실패가 최종 run 경로를 남기지 않고 재시도 가능한지 실패 테스트를 작성한다.
- [x] staging 디렉터리 작성 후 `os.replace`로 최종 run을 승격한다.
- [x] 관련 artifact·report 테스트를 통과시킨다.

### Task 4: protocol-driven signal rule

**Files:**
- Modify: `research/qlib_validation/metrics.py`
- Modify: `research/qlib_validation/report.py`
- Modify: `tests/research/test_qlib_metrics.py`
- Modify: `tests/research/test_qlib_report.py`

- [x] protocol threshold와 required basket 변경이 판정에 반영되는 실패 테스트를 작성한다.
- [x] signal rule schema를 검증하고 `signal_verdict`에 명시적으로 전달한다.
- [x] malformed protocol은 결과 대신 실패하도록 검증한다.

### Task 5: 장기 DB 통합 회귀와 전체 감사

**Files:**
- Modify: `tests/research/test_qlib_integration.py`

- [x] 260세션 이상의 Toss SQLite fixture로 실제 CLI replay·metrics 산출을 검증한다.
- [x] DB SHA-256 불변, Stage 2 inconclusive, 금지 실행 필드 부재를 확인한다.
- [x] 기본 환경 전체 pytest, 격리 Qlib pytest, Ruff, lock, 정적 금지 필드 검색을 실행한다.
- [x] 최종 diff와 기존 사용자 변경 보존 여부를 인라인 적대적 리뷰한다.
