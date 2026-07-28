# Qlib 적대적 리뷰 보완 설계

## 목적

Stage 1 Qlib 연구 실행이 동시 DB 갱신, 손상된 정책·시장 데이터, CLI 오입력,
부분 산출물 또는 설정 드리프트를 만났을 때 결과를 만들지 않고 실패 폐쇄한다.
운영 API·CLI·SQLite schema와 IPS 상태·주문 계약은 변경하지 않는다.

## 선택한 접근

기존 모듈 경계를 유지하면서 각 경계의 입력 계약을 강화한다.

- `source.py`는 하나의 명시적 SQLite read transaction에서 정책과 모든 candle을
  읽고, 저장 정책 해시·필수 벤치마크 4개·candle 무결성을 검증한다.
- `cli.py`는 argparse 오류를 포함한 모든 사용자 입력 오류를 stdout의 JSON 한
  객체와 비정상 종료 코드로 반환한다.
- `artifacts.py`는 실제 replay가 의존하는 정책 validator까지 source manifest에
  포함하고, 보고서 전체를 임시 run 디렉터리에 작성한 뒤 최종 경로로 원자적으로
  승격한다.
- `report.py`와 `metrics.py`는 `protocol.json.signal_rule`을 검증하고 실제 판정에
  전달한다. Stage 2는 계속 `inconclusive/stage2_not_run`이다.
- 장기 SQLite fixture로 source부터 CLI까지 200세션 이상 경로를 검증한다.

## 실패 처리

정책 해시 불일치, 벤치마크 identity 불일치, 비정상 candle, 데이터 중복·공백,
protocol 불일치는 bounded 예외로 종료한다. CLI는 예외 종류와 메시지만 JSON으로
반환한다. 실패한 run은 최종 run 경로를 점유하지 않는다.

## 비범위

Qlib factor 합성, 포트폴리오 NAV·환율·비용, 정책 활성화, IPS status 생성,
매매 방향·수량·가격·실행 필드는 추가하지 않는다.

## 검증

각 리뷰 finding마다 실패 회귀 테스트를 먼저 추가한다. 연구 환경 전체, 기본 환경
전체, Ruff, lock 검사를 통과하고 정적 금지 필드 검색 결과가 기존 허용 위치만
포함해야 완료다.
