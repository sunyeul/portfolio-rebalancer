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

성과 추적은 Toss 스냅샷 사이의 계좌 평가금과 원금 변동을 별도로 관리합니다. 최초 완료 스냅샷을 기준선으로 한 번 확인한 뒤, 이후 동기화마다 새 성과 포인트를 갱신합니다.

```bash
uv run ips-pilot performance baseline-preview --snapshot-id 4
uv run ips-pilot performance baseline-confirm --snapshot-id 4 --expected-principal-krw 120802745.17802304
uv run ips-pilot performance refresh
uv run ips-pilot performance candidates
uv run ips-pilot performance history --latest
```

설명되지 않은 현금 이동은 후보로 남으며 자동으로 입금·출금으로 확정하지 않습니다. 불완전·오래된·실패 스냅샷은 성과 포인트가 되지 않습니다.

## 저장소와 검증

기본 SQLite 경로는 `data/portfolio_rebalancer.sqlite3`입니다. 스키마는 Toss 계좌 관찰, 성과 추적, 계층/논지 프로필, 정책 버전만 보존합니다. 마이그레이션은 무결성 검사와 secure delete를 수행합니다.

```bash
uv run ruff format --check .
uv run ruff check .
uv run pytest -q
```

현재 구현은 Phase 2 기반까지 포함하며, 이후 단계에서 현금 정책·성과 설명·IPS 검토 화면을 확장할 수 있습니다. 로드맵과 Toss 전용 수렴 설계는 [`docs/superpowers/specs/2026-07-23-toss-only-foundation-convergence-design.md`](docs/superpowers/specs/2026-07-23-toss-only-foundation-convergence-design.md)와 [`docs/superpowers/specs/2026-07-22-cash-account-observability-roadmap-design.md`](docs/superpowers/specs/2026-07-22-cash-account-observability-roadmap-design.md)에 있습니다.

