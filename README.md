# 포트폴리오 리밸런서

FastAPI 계산 백엔드와 Bun/Vite/React 프론트엔드로 구성된 포트폴리오 리밸런싱 워크벤치입니다. 티커와 비중을 입력하면 가격 데이터를 조회하고, 위험기여도/효율점수/IPS 기반 실행 제안을 계산합니다.

## Stack

- Backend: FastAPI, pandas, numpy, scipy, yfinance, Pydantic
- Frontend: Bun, Vite, React, TypeScript
- Frontend libraries: TanStack Query, TanStack Table, React Hook Form, Zod, PapaParse, Recharts, lucide-react

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

- `POST /api/v1/portfolio/manual`
- `POST /api/v1/portfolio/csv`
- `POST /api/v1/analysis/run`
- `POST /api/v1/evaluation/run`
- `GET /api/v1/evaluation/download-csv`

세션은 서명된 `session_id` cookie로 유지됩니다.

## Docker

```bash
docker compose up --build
```

Docker 빌드는 Bun으로 프론트를 빌드한 뒤, Python 런타임 이미지에서 FastAPI가 빌드 산출물을 서빙합니다.

## 주의

이 앱은 교육/프로토타이핑 목적입니다. 계산 결과는 투자 자문이 아니며, 데이터는 `yfinance` 공급 상태와 네트워크에 영향을 받을 수 있습니다.
