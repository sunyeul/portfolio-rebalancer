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
rtk uv run --project research/qlib_validation python -m research.qlib_validation.cli stage1 --db data/portfolio_rebalancer.sqlite3 --as-of 2026-07-28T00:00:00+00:00 --output research/qlib_validation/artifacts --universe current-holdings
```

출력은 연구 근거이며 IPS 상태, 정책 활성화 또는 주문 권한이 아니다. 종목별로
21개 이상의 적격 일봉이 없으면 팩터를 만들지 않으며 capability는
`factor_unavailable`로 닫힌다. adapter 통과는 Qlib 모델 훈련, Stage 2 백테스트,
목표 정책 결론 또는 매매 권한을 뜻하지 않는다.

`--universe current-holdings`는 최신의 완전한 Toss 계좌 스냅샷에서 실제로
보유 중이면서 활성 정책에도 있는 종목만 연구 묶음에 넣는다. 빠진 정책 종목은
`research-universe.json`에 기록할 뿐, 계좌·정책·평가 결과를 수정하지 않는다.

20거래일 연구 예측:

```bash
rtk uv run --project research/qlib_validation python -m research.qlib_validation.cli forecast --db data/portfolio_rebalancer.sqlite3 --as-of 2026-07-28T00:00:00+00:00 --output research/qlib_validation/artifacts --universe current-holdings
```

예측은 Qlib `StaticDataLoader`를 통과한 입력에 Qlib 네이티브 `LGBModel`을
적용한다. LightGBM의 학습 횟수는 바깥 holdout보다 앞선 구간에서 20 거래일
간격을 둔 안쪽 검증으로만 고른다. 예측 대상은 종목별 현지 통화 종가 기준의
20거래일 수익률이며, 환율·계좌 수익률·정책 변경을 포함하지 않는다.
Qlib의 실험 기록은 실행 중에만 존재하는 임시 SQLite 저장소에 격리한다.
`holdout-predictions.json`의 20거래일 결과는 날짜별로 겹칠 수 있으므로 행 수를
독립 표본 수처럼 해석하지 않는다.
