from research.qlib_validation.replay import replay_regimes


def test_replay_uses_each_markets_last_same_month_candle_and_no_future_rows(
    long_snapshot,
):
    seen = []

    def evaluator(series_by_key, *, active_policy, last_change_at, now):
        seen.append((series_by_key, now))
        return {"regime": "risk_off", "reason": "regime_target_change"}

    points = replay_regimes(long_snapshot, evaluator=evaluator)
    assert points
    assert all(point.regime == "risk_off" for point in points)
    for series_map, decision_timestamp in seen:
        for rows in series_map.values():
            assert len(rows) >= 200
            assert (
                max(row["available_at"] for row in rows)
                <= decision_timestamp.isoformat()
            )


def test_replay_output_does_not_copy_ips_status(long_snapshot):
    points = replay_regimes(long_snapshot)
    assert all("status" not in point.record() for point in points)
