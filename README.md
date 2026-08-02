# IPS Pilot

IPS Pilot은 Toss증권 계좌의 읽기 전용 관찰 스냅샷을 기반으로 현금·보유자산·성과 추이를 점검하는 CLI입니다. 출력은 관찰 및 검토 신호이며, 주문 지시나 자동 매매 기능이 아닙니다.

## 제품 경계

- 유일한 원천은 Toss Open API에서 정규화한 계좌 스냅샷입니다.
- 보유 종목, 현금, 원가, 현재가, 주문·체결 사실은 Toss 관찰 데이터에서만 취급합니다.
- 수동 포트폴리오 업로드, CSV/TSV 입력, yfinance, 일본 계좌용 대체 경로는 지원하지 않습니다.
- OAuth·계좌 조회·잔고/보유/환율/주문 내역 조회만 허용하며 주문 생성·변경·취소·수량 산정·체결은 제품 범위 밖입니다.
- IPS 분류는 `core`, `satellite`, `experiment` 계층과 `OK`, `Watch`, `Review`, `Action` 어휘를 유지합니다. `Action`은 예외 개입 가능성을 점검하라는 뜻이지 거래 허가가 아닙니다.

## 설정

실제 인증 정보는 `.env` 또는 셸 환경에만 설정합니다.

```bash
TOSS_OPEN_API_CLIENT_ID=...
TOSS_OPEN_API_CLIENT_SECRET=...
TOSS_OPEN_API_ACCOUNT_SEQ=...
```

`TOSS_OPEN_API_ACCOUNT_SEQ`는 계좌 검색 결과에서 사용할 Toss 계좌의 순번입니다. `toss-health`가 자격 증명, OAuth, 계좌 검색, 설정한 계좌 일치를 모두 확인합니다. 인증 정보와 액세스 토큰은 소스, SQLite, 로그, 브라우저 저장소, 테스트 fixture에 저장하지 않습니다.

## 실행

```bash
uv sync
task toss-health
task toss-sync
uv run ips-pilot toss-snapshots --latest
uv run ips-pilot policy show --active
uv run ips-pilot inspection run
bun install --cwd frontend
bun run --cwd frontend build
task toss-dashboard-api
```

모든 CLI 명령은 stdout에 JSON 객체 하나만 출력합니다. `toss-sync`는 Toss에서 보유 종목, KRW/USD 예수금, USD/KRW 환율, 종료 주문을 읽어 불변 관찰 스냅샷으로 저장합니다. 불완전·오래된·실패 스냅샷은 진단 증거로만 남고 최신 평가 가능한 스냅샷을 대체하지 않습니다.

## 계좌 관찰과 정책 레이어

검사는 활성 IPS 정책의 `instruments[].layer`만 읽어 `core`·`satellite`·`experiment` 비중을 평가합니다. 정책에 없는 종목에는 임의의 계층이나 상태를 추론하지 않습니다.

```bash
uv run ips-pilot policy template > toss-policy-template.json
uv run ips-pilot policy activate --file toss-policy.json --expected-current-version 1
```

정책에 없는 보유 종목이나 목표가 완성되지 않은 정책 종목은 `classification_coverage`에서 제외되며 비중 판정은 `not_evaluable`로 닫힙니다.

## 성과 추적

성과 추적은 Toss 스냅샷 사이의 보유 종목 매입원가와 투자자산 평가금을 별도로 관리합니다. 투자 원금은 각 스냅샷의 보유 종목 매입원가이며, 손익과 보유 수익률은 투자자산 평가금과 매입원가의 차이로 계산합니다. 외부 순입출금 분류와 기준선은 TWR 구간을 나누는 계좌 성과 근거로만 유지합니다.

```bash
uv run ips-pilot performance baseline-preview --snapshot-id 4
# 기준선 원금은 TWR·외부 현금흐름 분류용으로만 확인
CONFIRMED_PRINCIPAL_KRW=...
uv run ips-pilot performance baseline-confirm --snapshot-id 4 --expected-principal-krw "$CONFIRMED_PRINCIPAL_KRW"
uv run ips-pilot performance refresh
uv run ips-pilot performance candidates
```

설명되지 않은 현금 이동은 후보로 남으며 자동으로 입금·출금으로 확정하지 않습니다. 불완전·오래된·실패 스냅샷은 성과 포인트가 되지 않습니다.
실패·부분 동기화가 새로 들어오면 이전 완료 스냅샷은 `last_verified_complete`로만 남고 현재 평가 대상으로 승격되지 않습니다.
Overview의 기본 흐름은 `매입원가 → 투자자산 평가금 → 보유 수익률`입니다. 보유 손익은 `투자자산 평가금 - 매입원가`, 보유 수익률은 그 손익을 매입원가로 나눈 값입니다. 연간 목표 10%는 별도의 YTD 계좌 TWR 지표로 관리하며, 현금 포함 계좌 성과와 매입원가 기준 보유 성과를 섞지 않습니다.

## 정책과 운영 검사

현금 리저브는 총계좌 평가금 기준 3% 최소·5% 중립 목표·10% 최대 범위로 관찰합니다. 중립 레이어 목표는 투자금 평가금 기준 Core 60%·Satellite 38%·Experiment 2%이며, 시장 국면에 따라 승인된 범위 안에서 함께 조정할 수 있습니다. 연간 목표 수익률은 누적 수익률과 분리한 연초 기준 YTD TWR 10%이며, 최근 1년 TWR은 보조 뷰로 함께 표시합니다.

```bash
uv run ips-pilot policy template > toss-policy-template.json
uv run ips-pilot policy validate --file toss-policy-template.json
uv run ips-pilot policy activate --file toss-policy.json --expected-current-version 1
uv run ips-pilot inspection run
# 터미널 1: API와 빌드된 화면
task toss-dashboard-api
# 개발 중 자동 새로고침이 필요하면 터미널 2
task toss-dashboard-dev
```

개발 화면은 `http://127.0.0.1:5173`, 빌드된 운영 화면은
`http://127.0.0.1:8000`에서 확인합니다. `task toss-dashboard-build`는
Bun으로 의존성을 고정 설치하고 `frontend/dist`를 다시 생성합니다.

정책 파일은 앱이 관리하는 목표·범위·레이어만 담습니다. Toss에서 관찰하지 않은 종목, 현재 보유 종목의 미분류 상태, 목표 합계 오류는 활성화할 수 없습니다. `inspection` 결과는 `OK`, `Watch`, `Review`, `Action`만 사용하며, `Action`도 예외 개입 가능성을 사람이 점검하라는 뜻입니다. 주문 수량이나 실행 플래그는 제공하지 않습니다.

### 종목 drawdown 근거

`market sync`는 활성 정책 종목(또는 `--symbols`로 지정한 종목)의 Toss 조정 일봉만 저장합니다. 이 데이터는 종목별 drawdown 검사에만 사용하며 시장 국면·기술지표·정책 후보를 만들지 않습니다.

```bash
uv run ips-pilot market sync
```

### Phase 5-v2 결과 계약

검사 결과는 `allocation_state`, `status`, `priority`, `suggestion`을
분리합니다. `allocation_state`는 `complete`, `partial`,
`not_evaluable` 중 하나이며, 유효한 전액 현금 계좌는 현금만 판정하는
`partial`입니다. Toss 원천이 오래되었거나 불완전·불일치하면
`not_evaluable`로 닫고, 차단 큐 항목에 우선순위와 제안을 붙이지 않습니다.

`status`는 `OK`/`Watch`/`Review`/`Action`의 심각도이고, `priority`는
`P1`(다음 정기매수 전), `P2`(이번 월간 점검), `P3`(다음 정기매수 배분
반영), `P4`(관찰 유지)의 재검토 시점입니다. `suggestion`은 닫힌 검토
코드만 사용하며, 주문 방향·수량·실행을 뜻하지 않습니다.

Overview는 `adjustment_suggestions`(현금·레이어·종목의 비중 조정 정책
검토)를 먼저 보여주고, 현금/투자자산 비중 밴드와 투자 원금·현재 평가금·
수익률을 보조 정보로 표시합니다. 성과 데이터가 없더라도 현재 Toss
스냅샷의 비중 판정은 계속할 수 있으며, YTD 목표 수익률 미달·수익·손실·
drawdown만으로 `Action`이나 거래 제안을 만들지 않습니다. 현금은 총계좌
기준 전략 자산이고 레이어·종목은 투자자산 기준으로 계산합니다.

Phase 5 손익·예외 검토는 제안 정책을 먼저 읽기 전용으로 확인하는 preview를 지원합니다.

```bash
uv run ips-pilot inspection preview --policy-file docs/superpowers/specs/2026-07-23-pattern-b-policy-draft.json
```

Preview는 정책을 활성화하거나 평가 행을 저장하지 않습니다. 현재 평가에
필요한 종목 레이어는 정책의 `instruments[].layer`에서만 읽습니다. `Action`은 자동 주문이 아니라 사람이
예외 개입 가능성을 확인하는 검사 상태입니다. 시장 동기화, 레이어 분류
갱신, 정책 활성화, 최초 `phase5-v2` 평가 저장은 운영자가 별도로 승인한
뒤 수행합니다. 이전 엔진 버전의 저장 결과는 새 계약으로 추정 변환하지
않고 명시적인 계약 불일치로 표시합니다.

## 저장소와 검증

기본 SQLite 경로는 `data/portfolio_rebalancer.sqlite3`입니다. 스키마는 Toss 계좌 관찰, 성과 추적, 정책 버전, 시장 근거만 보존합니다. 마이그레이션 후에는 무결성을 검사합니다.

```bash
uv run ruff format --check .
uv run ruff check .
uv run pytest -q tests --ignore=tests/research
```

현재 구현은 Phase 0–4의 Toss 전용 관찰·대시보드와 Phase 5의 손익·drawdown·예외 검토 및 `phase5-v2` 결과-first 비중 조정 계약까지 포함합니다. 인증된 의도 편집과 사람의 결정 기록은 아직 구현하지 않습니다. 승인된 Pattern B 정책은 [`docs/superpowers/specs/2026-07-23-pattern-b-policy-draft.md`](docs/superpowers/specs/2026-07-23-pattern-b-policy-draft.md)에 기록되어 있습니다.
