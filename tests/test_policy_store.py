import sqlite3

from storage.database import initialize_database
from storage.policy_store import DEFAULT_POLICY, get_active_policy, policy_hash


def test_default_policy_is_seeded_once_and_is_replayable(monkeypatch, tmp_path):
    path = tmp_path / "policy.sqlite3"
    monkeypatch.setenv("PORTFOLIO_DB_PATH", str(path))

    initialize_database()
    first = get_active_policy()
    initialize_database()
    second = get_active_policy()

    assert first is not None
    assert second == first
    assert first["version"] == 1
    assert first["policy"] == DEFAULT_POLICY
    assert first["policy_hash"] == policy_hash(DEFAULT_POLICY)
    with sqlite3.connect(path) as conn:
        assert (
            conn.execute("SELECT COUNT(*) FROM ips_policy_versions").fetchone()[0] == 1
        )
