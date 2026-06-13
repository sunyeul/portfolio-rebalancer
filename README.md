# IPS Pilot

IPS Pilot은 개인 투자 운영 원칙(IPS)을 기준으로 포트폴리오 상태를 진단하고, 다음 정기매수 조정안과 투자 논리 점검 항목을 제안하는 앱/CLI 워크벤치입니다. 티커와 비중을 입력하면 가격 데이터를 조회하고, 위험기여도/효율점수/IPS 기반 DCA Plan과 Review Queue를 계산합니다. 저장된 포트폴리오와 스냅샷을 SQLite에 보관해 월간 점검 흐름을 이어서 다룰 수 있습니다.

## Stack

- Backend: FastAPI, pandas, numpy, scipy, yfinance, Pydantic
- Frontend: Bun, Vite, React, TypeScript
- Frontend libraries: TanStack Query, TanStack Table, React Hook Form, Zod, PapaParse, Recharts, lucide-react
- Storage: SQLite

## 주요 기능

- 포트폴리오 붙여넣기, 수동 테이블 입력, CSV/TSV 업로드
- IPS 그룹 분류: `core`, `satellite_ai_infra`, `satellite_ai_software`, `satellite_nextgen`
- 기간별 가격 데이터 조회와 포트폴리오/벤치마크/자산별 지표 계산
- 기본 무위험 수익률 2.5%, 현실 벤치마크 `SPY:80,QQQ:20` 지원
- 위험기여도, 효율점수, 히스테리시스, 최소 거래 기준을 반영한 평가
- DCA 증액/감액, 투자 논리 점검, 예외적 매도 검토 등 IPS 액션 분류
- 저장된 포트폴리오, 현재 상태 자동 저장, 스냅샷 생성/편집/삭제/로드
- IPS 목표 비중과 룰 편집

## 실행

```bash
# Python 의존성
uv sync

# Frontend 의존성
cd frontend
bun install
cd ..

# API 서버
make run

# 별도 터미널에서 Vite 개발 서버
make frontend-dev
```

개발 중에는 Vite가 `/api` 요청을 `http://localhost:8000`으로 프록시합니다. 프로덕션 빌드는 FastAPI가 `frontend/dist`를 `/`로 서빙합니다.

기본 SQLite DB는 `data/portfolio_rebalancer.sqlite3`에 생성됩니다. 경로를 바꾸려면 `PORTFOLIO_DB_PATH` 환경 변수를 설정합니다.

## 주요 명령어

```bash
make run             # FastAPI API 서버
make frontend-dev    # Vite 개발 서버
make dev             # API + 프론트 개발 서버
make build-frontend  # React 앱 빌드
uv run pytest        # 백엔드 테스트
```

## Codex용 CLI

CLI는 사람이 보는 리포트보다 Codex 같은 에이전트가 안정적으로 파싱하는 JSON 출력을 기본으로 합니다. stdout에는 단일 JSON 객체만 출력하고, 웹앱과 같은 `PORTFOLIO_DB_PATH`/SQLite DB와 IPS 설정을 공유합니다.

```bash
uv run ips-pilot evaluate --text "VOO 40
QQQ 60"
uv run ips-pilot evaluate --file portfolio.csv --output-dir out
uv run ips-pilot portfolios list
uv run ips-pilot snapshots list --portfolio-id 1
uv run ips-pilot evaluate --snapshot-id 14
uv run ips-pilot evaluate --portfolio-id 1 --save
```

기존 자동화와 리포트 호환성을 위해 `uv run portfolio-rebalancer ...` 명령도 계속 지원합니다.

`--portfolio-id`는 웹앱의 current-state를 읽고, `--snapshot-id`는 저장된 스냅샷을 읽습니다. DB 저장은 `--save` 또는 `--save-to-portfolio-id`를 명시한 경우에만 수행합니다.

Codex에게 사용을 맡길 때는 먼저 포트폴리오와 스냅샷 목록을 확인하게 한 뒤, 원하는 스냅샷을 평가하게 하면 됩니다.

```bash
uv run ips-pilot portfolios list
uv run ips-pilot snapshots list --portfolio-id 1
uv run ips-pilot evaluate --snapshot-id 14 --output-dir /tmp/ips_pilot_eval_14
```

평가 JSON에서는 DCA Plan과 Review Queue를 만들기 위해 `agent_summary.recommended_actions`, `agent_summary.hold_actions`, `agent_summary.data_quality_warnings`, `analysis.portfolio_metrics`, `evaluation.proposal`을 우선 읽으면 됩니다. `--output-dir`를 지정하면 같은 결과가 `metrics.csv`, `proposal.csv`, `ips_actions.csv`, `group_summary.csv`, `rc_violations.csv`로도 저장됩니다.

기본 설정은 샤프비율과 초과수익률 계산에 맞춰 무위험 수익률 `2.5%`, 벤치마크 `SPY:80,QQQ:20`입니다. 단순 성과 표시만 비교할 때는 `--rf 0` 또는 UI 입력값 `0`도 사용할 수 있습니다.

프론트 검증:

```bash
cd frontend
bun run typecheck
bun run build
```

## API

모든 애플리케이션 API는 `/api/v1` 아래에 있습니다.

워크벤치:

- `POST /api/v1/portfolio/manual`
- `POST /api/v1/portfolio/csv`
- `POST /api/v1/analysis/run`
- `POST /api/v1/evaluation/run`
- `GET /api/v1/evaluation/download-csv`

저장된 포트폴리오와 스냅샷:

- `GET /api/v1/portfolios`
- `POST /api/v1/portfolios`
- `PATCH /api/v1/portfolios/{portfolio_id}`
- `GET /api/v1/portfolios/{portfolio_id}/current-state`
- `POST /api/v1/portfolios/{portfolio_id}/current-state`
- `GET /api/v1/portfolios/{portfolio_id}/snapshots`
- `POST /api/v1/portfolios/{portfolio_id}/snapshots`
- `GET /api/v1/portfolios/snapshots/{snapshot_id}`
- `POST /api/v1/portfolios/snapshots/{snapshot_id}/load`
- `PATCH /api/v1/portfolios/snapshots/{snapshot_id}`
- `DELETE /api/v1/portfolios/snapshots/{snapshot_id}`

설정:

- `GET /api/v1/config/options`
- `GET /api/v1/config/ips`
- `PUT /api/v1/config/ips/target-allocations`
- `PUT /api/v1/config/ips/rules`
- `PUT /api/v1/config/ips/action-priorities` (읽기 전용 정책으로 403 반환)

세션은 서명된 `session_id` cookie로 유지됩니다.

## 입력 형식

붙여넣기는 티커와 비중을 기본으로 받습니다.

```text
VOO 40
QQQ 25
SOXX 15
```

CSV/TSV와 수동 테이블은 다음 필드를 지원합니다.

- `ticker`: 자산 티커
- `allocation`: 배분 비율, 단위 %
- `return_total`: 누적 수익률 override, 단위 %
- `group`: `core`, `satellite_ai_infra`, `satellite_ai_software`, `satellite_nextgen` 중 하나
- `dca_enabled`: 정기매수 조정 대상 여부
- `thesis_status`: `unknown`, `intact`, `watch`, `broken`

지원하지 않는 그룹은 `core`로 정규화됩니다. 붙여넣기 화면에서는 IPS 그룹만 먼저 선택하고, DCA와 투자 논리 상태는 분석 후 세부 판단값에서 보정합니다.

## IPS 설정

IPS 목표 비중은 네 그룹별로 `min <= target <= max` 범위로 저장합니다. 현재 판단 로직은 `core`를 코어 역할로, `satellite_ai_infra`, `satellite_ai_software`, `satellite_nextgen`을 위성 역할로 접어 코어/위성 상태를 계산합니다. 액션 우선순위는 앱 정의값을 읽기 전용으로 노출하며, 룰 JSON은 설정 화면 또는 API로 교체할 수 있습니다.

## Docker

```bash
docker compose up --build
```

Docker 빌드는 Bun으로 프론트를 빌드한 뒤, Python 런타임 이미지에서 FastAPI가 빌드 산출물을 서빙합니다.

## 주의

이 앱은 교육/프로토타이핑 목적입니다. 계산 결과는 투자 자문이 아니며, 데이터는 `yfinance` 공급 상태와 네트워크에 영향을 받을 수 있습니다.
