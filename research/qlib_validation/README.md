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
