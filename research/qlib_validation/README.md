# Qlib validation research

Qlib 설치와 API 설명은 [공식 stable 문서](https://qlib.readthedocs.io/en/stable/)를 기준으로 한다.
이 디렉터리는 `pyqlib==0.9.7` lock, IPS Pilot 전용 입력 계약과 재현 가능한 실행만 소유한다.

환경 확인:

```bash
rtk uv run --project research/qlib_validation python -m research.qlib_validation.environment
```

Qlib 연구 입력 보충:

```bash
uv run ips-pilot market sync --research-only --target-points 756 --max-pages 4
```

이 명령은 공식 Toss 일봉만 정규화하여 불변 market candle 저장소에 추가한다.
`--research-only`에서는 동적 배분 평가와 정책 후보 저장을 호출하지 않는다.
Qlib 입력을 만들 때는 각 시계열의 현재 종가와 20 거래일 전 종가만 사용한
후행 수익률 팩터를 메모리에서 파생하며, warm-up 20개 행은 제외한다. 원본 DB에는
팩터를 쓰지 않고 미래 시세도 사용하지 않는다.

Stage 1 실행:

```bash
rtk uv run --project research/qlib_validation python -m research.qlib_validation.cli stage1 --db data/portfolio_rebalancer.sqlite3 --as-of 2026-07-28T00:00:00+00:00 --output research/qlib_validation/artifacts
```

출력은 연구 근거이며 IPS 상태, 정책 활성화 또는 주문 권한이 아니다. 종목별로
21개 이상의 적격 일봉이 없으면 팩터를 만들지 않으며 capability는
`factor_unavailable`로 닫힌다. adapter 통과는 Qlib 모델 훈련, Stage 2 백테스트,
목표 정책 결론 또는 매매 권한을 뜻하지 않는다.
