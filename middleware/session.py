"""세션 관리 미들웨어.

# AIDEV-NOTE: session-management; Redis 기반 세션 (개발 환경에서는 메모리 기반 폴백)
"""

import os
import uuid
from typing import Any

import pandas as pd
from itsdangerous import BadSignature, Signer


class SessionManager:
    """세션 관리자.

    개발 환경에서는 메모리 기반 딕셔너리를 사용하고,
    프로덕션 환경에서는 Redis를 사용합니다.
    """

    def __init__(self, secret_key: str | None = None):
        """세션 관리자 초기화.

        Args:
            secret_key: 세션 서명용 비밀 키 (None이면 환경 변수에서 읽음)
        """
        self.secret_key = secret_key or os.getenv(
            "SESSION_SECRET_KEY", "dev-secret-key"
        )
        self.signer = Signer(self.secret_key)
        self._memory_store: dict[str, dict[str, Any]] = {}

    def create_session_id(self) -> str:
        """새 세션 ID를 생성합니다.

        Returns:
            세션 ID 문자열
        """
        return str(uuid.uuid4())

    def sign_session_id(self, session_id: str) -> str:
        """세션 ID에 서명을 추가합니다.

        Args:
            session_id: 세션 ID

        Returns:
            서명된 세션 ID
        """
        return self.signer.sign(session_id).decode()

    def unsign_session_id(self, signed_session_id: str) -> str | None:
        """서명된 세션 ID에서 원본을 추출합니다.

        Args:
            signed_session_id: 서명된 세션 ID

        Returns:
            원본 세션 ID 또는 None (서명 검증 실패 시)
        """
        try:
            return self.signer.unsign(signed_session_id).decode()
        except BadSignature:
            return None

    def get(self, session_id: str, key: str, default: Any = None) -> Any:
        """세션에서 값을 가져옵니다.

        Args:
            session_id: 세션 ID
            key: 키
            default: 기본값

        Returns:
            저장된 값 또는 기본값
        """
        if session_id not in self._memory_store:
            return default
        return self._memory_store[session_id].get(key, default)

    def set(self, session_id: str, key: str, value: Any) -> None:
        """세션에 값을 저장합니다.

        Args:
            session_id: 세션 ID
            key: 키
            value: 값 (DataFrame은 JSON으로 직렬화)
        """
        if session_id not in self._memory_store:
            self._memory_store[session_id] = {}

        # DataFrame은 JSON으로 직렬화
        if isinstance(value, pd.DataFrame):
            value = value.to_dict(orient="records")
        elif isinstance(value, pd.Series):
            value = value.to_dict()

        self._memory_store[session_id][key] = value

    def get_dataframe(self, session_id: str, key: str) -> pd.DataFrame | None:
        """세션에서 DataFrame을 가져옵니다.

        Args:
            session_id: 세션 ID
            key: 키

        Returns:
            DataFrame 또는 None
        """
        data = self.get(session_id, key)
        if data is None:
            return None
        if isinstance(data, list):
            return pd.DataFrame(data)
        return data

    def get_series(self, session_id: str, key: str) -> pd.Series | None:
        """세션에서 Series를 가져옵니다.

        Args:
            session_id: 세션 ID
            key: 키

        Returns:
            Series 또는 None
        """
        data = self.get(session_id, key)
        if data is None:
            return None
        if isinstance(data, dict):
            return pd.Series(data)
        return data

    def delete(self, session_id: str, key: str) -> None:
        """세션에서 키를 삭제합니다.

        Args:
            session_id: 세션 ID
            key: 키
        """
        if session_id in self._memory_store:
            self._memory_store[session_id].pop(key, None)

    def clear(self, session_id: str) -> None:
        """세션을 완전히 삭제합니다.

        Args:
            session_id: 세션 ID
        """
        self._memory_store.pop(session_id, None)


# 전역 세션 관리자 인스턴스
session_manager = SessionManager()
