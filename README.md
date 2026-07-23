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
uv run ips-pilot account-view --snapshot-id 4
uv run ips-pilot policy show --active
uv run ips-pilot inspection run
uv run ips-pilot inspection show --latest
bun install --cwd frontend
bun run --cwd frontend build
task toss-dashboard-api
```

모든 CLI 명령은 stdout에 JSON 객체 하나만 출력합니다. `toss-sync`는 Toss에서 보유 종목, KRW/USD 예수금, USD/KRW 환율, 종료 주문을 읽어 불변 관찰 스냅샷으로 저장합니다. 불완전·오래된·실패 스냅샷은 진단 증거로만 남고 최신 평가 가능한 스냅샷을 대체하지 않습니다.

## 계좌 관찰과 프로필

`account-view`는 특정 완료 스냅샷을 현금 비중, 투자자산 비중, 계층별 비중, 분류 커버리지로 투영합니다. 분류되지 않은 종목에 임의의 계층이나 상태를 추론하지 않습니다.

```bash
uv run ips-pilot profiles list
uv run ips-pilot profiles set --symbol AAPL --market-country US --layer satellite --thesis-status valid --note "전략적 위성"
```

프로필은 이미 Toss 스냅샷에서 관찰된 종목에만 설정할 수 있고, 원본 브로커 관찰 행을 변경하지 않습니다. 프로필이 없는 종목은 `classification_coverage`에서 제외되며 추가 검토 대상으로 표시됩니다.

## 성과 추적

성과 추적은 Toss 스냅샷 사이의 투자 원금과 계좌 평가금을 별도로 관리합니다. 투자 원금은 확인된 초기 원금에 분류된 외부 순입출금만 반영하며, 현금과 종목 사이의 이동·매매·손익은 원금을 바꾸지 않습니다. 최초 완료 스냅샷을 기준선으로 한 번 확인한 뒤, 이후 동기화마다 새 성과 포인트를 갱신합니다.

```bash
uv run ips-pilot performance baseline-preview --snapshot-id 4
uv run ips-pilot performance baseline-confirm --snapshot-id 4 --expected-principal-krw 120802745.17802304
uv run ips-pilot performance refresh
uv run ips-pilot performance candidates
uv run ips-pilot performance history --latest
```

설명되지 않은 현금 이동은 후보로 남으며 자동으로 입금·출금으로 확정하지 않습니다. 불완전·오래된·실패 스냅샷은 성과 포인트가 되지 않습니다.
실패·부분 동기화가 새로 들어오면 이전 완료 스냅샷은 `last_verified_complete`로만 남고 현재 평가 대상으로 승격되지 않습니다.
Overview의 기본 흐름은 `투자 원금 → 계좌 평가금 → 원금 대비 계좌 수익률`입니다. 계좌 손익은 `계좌 평가금 - 투자 원금`, 원금 대비 계좌 수익률은 그 손익을 투자 원금으로 나눈 값이며, 원금 근거가 없으면 수익률을 산출하지 않습니다. 연간 목표 10%는 별도의 YTD 계좌 TWR 지표로 관리하고, 보유 종목의 매입원가 기준 평가손익률과 섞지 않습니다.

## 정책과 운영 검사

현금 리저브는 총계좌 평가금 기준 10% 최소·15% 목표·20% 최대 범위로 관찰합니다. 레이어와 종목은 투자금 평가금 기준으로 별도 목표 범위를 사용합니다. 연간 목표 수익률은 누적 수익률과 분리한 연초 기준 YTD TWR 10%이며, 최근 1년 TWR은 보조 뷰로 함께 표시합니다.

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

정책 파일은 앱이 관리하는 목표·범위·프로필 의도만 담습니다. Toss에서 관찰하지 않은 종목, 현재 보유 종목의 미분류 상태, 목표 합계 오류는 활성화할 수 없습니다. `inspection` 결과는 `OK`, `Watch`, `Review`, `Action`만 사용하며, `Action`도 예외 개입 가능성을 사람이 점검하라는 뜻입니다. 주문 수량이나 실행 플래그는 제공하지 않습니다.

Phase 5 손익·예외 검토는 제안 정책을 먼저 읽기 전용으로 확인하는 preview를 지원합니다.

```bash
uv run ips-pilot inspection preview --policy-file docs/superpowers/specs/2026-07-23-pattern-b-policy-draft.json
uv run ips-pilot profiles set --symbol NBIS --market-country US --layer satellite --thesis-status watch --overlap-status review --management-burden-status clear --holdability-status clear --etf-substitution-status review --note "위성 투자 논지 재검토" --review-factors-note "중복과 ETF 대체 가능성 확인"
```

Preview는 정책을 활성화하거나 평가 행을 저장하지 않습니다. `Action`은 동일 종목의 손상된 논지와 하드 최대 비중 위반이 함께 확인될 때의 예외 검토 신호일 뿐입니다. 시장 동기화, 구조화된 프로필 갱신, 정책 활성화, 최초 `phase5-v1` 평가 저장은 운영자가 별도로 승인한 뒤 수행합니다.

## 저장소와 검증

기본 SQLite 경로는 `data/portfolio_rebalancer.sqlite3`입니다. 스키마는 Toss 계좌 관찰, 성과 추적, 계층/논지 프로필, 정책 버전만 보존합니다. 마이그레이션은 무결성 검사와 secure delete를 수행합니다.

```bash
uv run ruff format --check .
uv run ruff check .
uv run pytest -q
```

현재 구현은 Phase 0–4의 Toss 전용 관찰·대시보드와 Phase 5의 손익·drawdown·예외 검토 신호까지 포함합니다. Phase 6(인증된 의도 편집과 사람의 결정 기록)은 다음 로드맵 단계입니다. 현재 로드맵은 [`docs/superpowers/specs/2026-07-22-cash-account-observability-roadmap-design.md`](docs/superpowers/specs/2026-07-22-cash-account-observability-roadmap-design.md)에 있으며, 승인된 Pattern B 정책은 [`docs/superpowers/specs/2026-07-23-pattern-b-policy-draft.md`](docs/superpowers/specs/2026-07-23-pattern-b-policy-draft.md)에 기록되어 있습니다.
