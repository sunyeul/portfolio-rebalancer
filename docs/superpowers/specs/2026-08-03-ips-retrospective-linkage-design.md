# IPS 회고 핵심 연결 계층 설계

## 목적

IPS 평가의 Review Queue 항목, 사용자의 최초 결정, 이후 Toss 관측과 성과 근거를 연결한다. 이 기능은 회고용 검사 기록이며 주문·가격·수량·자동 정책 변경 권한을 만들지 않는다.

## 불변 기록

`ips_retrospective_cases`는 최신 current 평가의 비차단 Queue 항목 하나와 최초 결정(`adopted`, `deferred`, `declined`)을 고정한다. 당시 Queue 항목 JSON과 평가 run을 보존해 사후 결과가 최초 근거를 덮어쓰지 못하게 한다.

`ips_retrospective_reviews`는 `1m`, `3m`, `12m` 관측과 인간 판정을 append-only revision으로 남긴다. 동일 horizon의 수정은 새 행을 추가하고 가장 큰 revision을 현재 판정으로 사용한다.

## 증거와 안전 경계

각 horizon은 결정 시점 + 30/90/365일 이후 첫 complete Toss 스냅샷을 사용한다. 원래 정책 버전으로 현금·레이어·종목의 비중과 목표 gap 변화를 계산하고, 가능할 때만 동일 구간의 계좌 TWR·최대 drawdown을 함께 기록한다. 역사 스냅샷은 회고 계산에서만 허용하며 현재 IPS 평가의 freshness 규칙을 완화하지 않는다.

체결 원천은 시장 식별자를 보장하지 않으므로 공개 증거에는 체결 건수와 비중 변화 여부만 넣고, 레이어별 체결 연결이 불가능하면 `partial`로 표기한다. 주문 방향·수량·가격은 어떤 CLI 응답에도 넣지 않는다.

판단 품질, 이행 충실도, 정책 판단은 사용자가 확정한다. 성과·손익·drawdown은 근거일 뿐 자동 판정이나 `Action` 생성 근거가 아니다. `review_flag`는 회고 목록에만 표시되며 정책 후보, 활성 정책, 현재 평가, Review Queue를 변경하지 않는다.
