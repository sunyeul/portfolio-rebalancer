# 포트폴리오 리밸런서

FastAPI 계산 백엔드와 Bun/Vite/React 프론트엔드로 구성된 포트폴리오 리밸런싱 워크벤치입니다. 티커와 비중을 입력하면 가격 데이터를 조회하고, 위험기여도/효율점수/IPS 기반 실행 제안을 계산합니다. 저장된 포트폴리오와 스냅샷을 SQLite에 보관해 입력, 분석, 평가 결과를 이어서 다룰 수 있습니다.

## Stack

- Backend: FastAPI, pandas, numpy, scipy, yfinance, Pydantic
- Frontend: Bun, Vite, React, TypeScript
- Frontend libraries: TanStack Query, TanStack Table, React Hook Form, Zod, PapaParse, Recharts, lucide-react
- Storage: SQLite

## 주요 기능

- 포트폴리오 붙여넣기, 수동 테이블 입력, CSV/TSV 업로드
- 고정 IPS 그룹 분류: `core`, `satellite`, `cash`, `unclassified`
- 기간별 가격 데이터 조회와 포트폴리오/벤치마크/자산별 지표 계산
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
- `group`: `core`, `satellite`, `cash`, `unclassified` 중 하나
- `dca_enabled`: 정기매수 조정 대상 여부
- `thesis_status`: `unknown`, `intact`, `watch`, `broken`

지원하지 않는 그룹은 `unclassified`로 정규화됩니다. 붙여넣기 화면에서는 IPS 그룹만 먼저 선택하고, DCA와 투자 논리 상태는 분석 후 세부 판단값에서 보정합니다.

## IPS 설정

IPS 목표 비중은 `core`와 `satellite`에 대해 `min <= target <= max` 범위로 저장합니다. 액션 우선순위는 앱 정의값을 읽기 전용으로 노출하며, 룰 JSON은 설정 화면 또는 API로 교체할 수 있습니다.

## Docker

```bash
docker compose up --build
```

Docker 빌드는 Bun으로 프론트를 빌드한 뒤, Python 런타임 이미지에서 FastAPI가 빌드 산출물을 서빙합니다.

## 주의

이 앱은 교육/프로토타이핑 목적입니다. 계산 결과는 투자 자문이 아니며, 데이터는 `yfinance` 공급 상태와 네트워크에 영향을 받을 수 있습니다.
