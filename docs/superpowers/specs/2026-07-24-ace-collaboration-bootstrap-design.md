# ACE 협업 운영 부트스트랩 설계

**상태:** 승인된 설계
**버전:** ACE Bootstrap v1
**범위:** IPS Pilot 저장소의 협업 운영 문서·메모리 계약·선택적 Codex 세션 훅

## 목적과 목표

- **PURPOSE:** 제품 안전 경계와 현재 지시를 우선하면서, 재사용 가능한 협업 교훈과 확인된 개인 선호를 안전하게 복원한다.
- **GOAL:** 수동으로 재현 가능한 ACE 계약과 Codex 세션 훅을 추가하고, 공유 교훈은 Git 추적 대상으로, 개인 선호는 로컬 전용으로 검증한다.
- **ALIGNMENT:** 제품 기능이나 투자 판단을 변경하지 않고, 기존 작업 트리의 미커밋 변경을 보존하면서 에이전트 협업의 재현성만 높인다.
- **WORKING LOG:** 기존 훅 설정은 없었다. 사용자가 훅 포함을 요청했으므로 프로젝트 훅을 구성하되 Codex의 명시적 검토·신뢰 절차를 유지한다. 자동 복원은 `SessionStart`의 `startup`, `resume`, `clear`, `compact`에서만 수행한다.

## 현재 저장소 제약

- 기존 `AGENTS.md`, `RTK.md`, `.gitignore`, `.codex/config.toml`은 유지하며 ACE 라우팅만 필요한 범위에서 덧붙인다.
- 작업 트리에 광범위한 사용자 변경이 있으므로 이 설계와 ACE 파일 외의 변경은 만들지 않는다.
- `uv run ruff format --check .`, `uv run ruff check .`, `uv run pytest -q`를 기본 검증으로 사용한다.
- `.serena/`는 이미 Git 무시 대상이다. `.serena/memories/local/user_preferences.md`를 로컬 선호 경로로 사용하고 중복 ignore 규칙은 추가하지 않는다.

## 구성 요소

### 1. `AGENTS.md` 라우팅

ACE 운영을 작업 시작, 범위·목표 변경, 명시적 선호 또는 정정 수신, 예상 밖의 재사용 가능한 실패, 메모리 후보가 있는 작업의 완료 직전에 호출한다. 현재 시스템·개발자·사용자 지시와 기존 저장소 계약이 ACE 항목보다 항상 우선임을 명시하고, 훅이 없거나 신뢰되지 않을 때의 수동 폴백 순서를 적는다.

### 2. `ace-collaboration-memory` 스킬

`.agent/skills/ace-collaboration-memory/SKILL.md`에 다음을 고정한다.

- 작업 종류·범위·사용 도구를 정하고 네 가지 작업 맥락을 선언한 뒤 관련 `Active` 교훈과 `Confirmed` 선호만 검색한다.
- 관찰을 `공유 교훈`, `명시적 선호`, `추론 선호`, `비메모리`로 분류한다.
- 메모리 변경은 안정적인 ID를 보존하는 `ADD`, `UPDATE`, `MARK`만 허용한다. 삭제, 전체 요약 재작성, ID 재사용은 금지한다.
- 명시적 선호는 즉시 `Confirmed`, 추론 선호는 사용자 확인 전 `Pending`으로만 기록한다.
- 재발을 관찰 가능한 검증으로 막을 수 있으면 기존 테스트·검증 소유자를 먼저 연결하고, 교훈 문구만으로 예방 완료를 주장하지 않는다.
- 비밀·자격 증명·원문 대화·대용량 명령 출력은 저장하지 않는다.

### 3. 공유 교훈 저장소

`.agent/playbooks/collaboration-lessons.md`는 Git 추적 파일이다. 각 항목은 `lesson-` 접두사의 안정 ID, `title`, `status`(`Active` 또는 `Superseded`), `scope`, 구체적인 `trigger`, 실행 가능한 `rule`, `evidence`, `helpful`, `harmful`을 가진다.

### 4. 로컬 사용자 선호 저장소

`.serena/memories/local/user_preferences.md`는 Git 무시 파일이다. 각 항목은 `pref-` 접두사의 안정 ID, `title`, `status`(`Confirmed`, `Pending`, `Rejected`, `Superseded`), `context`, `prefer`, `avoid`, `source`, `helpful`, `harmful`을 가진다. 템플릿만 추적되는 문서에는 개인 정보나 추정 선호를 넣지 않는다.

### 5. Codex 세션 훅

`.codex/hooks.json`에 프로젝트 로컬 `SessionStart` 훅을 하나 등록하고, 구현은 `.codex/hooks/ace_session_start.py`에 둔다. 기존 `.codex/config.toml`에 인라인 훅을 섞지 않는다.

훅은 다음 입력만 사용한다.

- 이벤트 matcher: `startup|resume|clear|compact`
- 세션 입력의 `cwd`와 `hook_event_name`
- Git 무시 로컬 선호 파일의 제한된 크기 내 내용

출력에는 항상 다음 고정 복원 지시를 포함한다.

1. `AGENTS.md`와 ACE 스킬을 읽는다.
2. 현재 `PURPOSE`, `GOAL`, `ALIGNMENT`, `WORKING LOG`를 선언한다.
3. 작업과 관련된 `Active` 공유 교훈만 검색한다.
4. 로컬 선호에서는 `Confirmed`만 적용하고, `Pending`·`Rejected`·`Superseded`는 적용하지 않는다.
5. 우선순위는 현재 지시 > 프로젝트 계약 > 활성 공유 교훈 > 확인된 개인 선호다.

선호 파일이 없거나 형식이 잘못되거나 최대 입력·출력 크기를 넘으면 선호를 전혀 출력하지 않고 짧은 경고만 추가한 뒤 종료 코드 0을 반환한다. 훅은 네트워크·파일 쓰기·메모리 변경을 하지 않으며, 공유 교훈 전체나 대화 원문을 자동 주입하지 않는다. Codex가 프로젝트 훅을 신뢰하기 전에는 수동 폴백이 유효하다.

## 데이터 흐름

```text
Codex SessionStart
  -> ace_session_start.py
  -> stdin의 cwd/event 검증
  -> 로컬 Confirmed 선호 전체 검증
  -> 고정 ACE 지시 + 제한된 Confirmed 선호 출력
  -> 에이전트가 AGENTS/SKILL 읽기 및 관련 Active lesson 검색
```

훅이 실패해도 세션은 중단되지 않는다. 수동 폴백은 `AGENTS.md`와 스킬을 읽고, `rg`로 관련 `Active` 교훈을 찾고, 선호 파일에서 유효한 `Confirmed` 항목만 읽는 순서를 그대로 따른다.

## 검증 계획

1. `git check-ignore`로 로컬 선호 파일이 무시되고 공유 교훈 파일은 무시되지 않는지 확인한다.
2. 훅 스크립트를 임시 stdin으로 실행해 `Confirmed`만 출력되는지, 다른 상태가 제외되는지, 누락·오류·크기 초과에서 종료 코드 0인지 확인한다.
3. JSON 훅 설정의 이벤트와 명령 경로를 정적으로 검사한다.
4. 기본 Ruff·pytest 검증을 실행한다. 실패하면 완료를 주장하지 않는다.
5. 최종 보고에 변경 파일, 훅 설정 및 신뢰 상태, 수동 폴백 증거, 검증 결과, 생성·갱신한 메모리 항목을 기록한다.

## 비범위와 안전 경계

- 제품 코드, API 계약, CLI 출력, 투자 판단, 주문·실행 기능은 변경하지 않는다.
- 스냅샷·설정·저널·SQLite 상태는 수정하지 않는다.
- 훅 신뢰를 자동 승인하거나 사용자 전역 설정을 수정하지 않는다.
- ACE 파일은 비밀, 토큰, 자격 증명, 원문 대화, 대용량 실행 출력을 저장하지 않는다.
