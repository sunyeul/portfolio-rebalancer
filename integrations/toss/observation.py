"""Normalize read-only Toss account observations into immutable snapshot data."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from hashlib import sha256
from typing import Any, Callable, Mapping

from integrations.toss.auth import TossAuthorizedReader
from integrations.toss.config import TossApiConfig


ACCOUNT_ALIAS = "toss-brokerage"
TOLERANCE = Decimal("0.01")


class SyncState(StrEnum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    STALE = "stale"
    FAILED = "failed"


class ObservationError(RuntimeError):
    """Raised when a Toss response cannot be normalized safely."""


@dataclass(frozen=True)
class NormalizedHolding:
    symbol: str
    name: str
    market_country: str
    currency: str
    quantity: float
    last_price: float
    average_purchase_price: float
    market_value_native: float
    market_value_krw: float | None
    cost_native: float
    cost_krw: float | None
    profit_loss_native: float
    profit_loss_krw: float | None
    daily_profit_loss_native: float
    daily_profit_loss_krw: float | None


@dataclass(frozen=True)
class NormalizedCash:
    currency: str
    buying_power_native: float
    buying_power_krw: float | None


@dataclass(frozen=True)
class NormalizedFxRate:
    base_currency: str
    quote_currency: str
    rate: float
    mid_rate: float | None
    valid_from: str | None
    valid_until: str | None


@dataclass(frozen=True)
class NormalizedOrder:
    order_id: str
    symbol: str
    currency: str
    side: str
    order_type: str
    status: str
    ordered_at: str | None
    canceled_at: str | None
    quantity: float
    order_price_native: float | None
    order_amount_native: float | None
    filled_quantity: float
    average_filled_price_native: float | None
    filled_amount_native: float | None
    commission_native: float | None
    tax_native: float | None
    filled_at: str | None
    settlement_date: str | None


@dataclass(frozen=True)
class NormalizedSnapshot:
    account_alias: str
    sync_started_at: str
    synced_at: str
    state: SyncState
    holdings: tuple[NormalizedHolding, ...]
    cash: tuple[NormalizedCash, ...]
    fx_rate: NormalizedFxRate | None
    orders: tuple[NormalizedOrder, ...]
    source_timestamps: dict[str, str | None]
    data_quality: dict[str, Any]
    reconciliation: dict[str, Any]
    total_value_krw: float | None
    invested_value_krw: float | None
    cash_value_krw: float | None
    fingerprint: str

    @property
    def cash_by_currency(self) -> dict[str, float]:
        return {item.currency: item.buying_power_native for item in self.cash}


def _decimal(value: Any, field: str, *, nonnegative: bool = False) -> Decimal:
    if value is None or isinstance(value, bool):
        raise ObservationError(f"invalid {field}")
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ObservationError(f"invalid {field}") from exc
    if not parsed.is_finite() or (nonnegative and parsed < 0):
        raise ObservationError(f"invalid {field}")
    return parsed


def _optional_decimal(value: Any, field: str) -> Decimal | None:
    if value is None:
        return None
    return _decimal(value, field)


def _float(value: Decimal | None) -> float | None:
    return None if value is None else float(value)


def _now_iso(clock: Callable[[], float]) -> str:
    return datetime.fromtimestamp(clock(), timezone.utc).isoformat()


def _market_amount(item: Mapping[str, Any], key: str, field: str) -> Decimal:
    value = item.get(key)
    if not isinstance(value, Mapping):
        raise ObservationError(f"missing {field}")
    return _decimal(value.get("amount"), field)


def _purchase_amount(item: Mapping[str, Any], field: str) -> Decimal:
    value = item.get("cost")
    if not isinstance(value, Mapping):
        raise ObservationError(f"missing {field}")
    return _decimal(value.get("purchaseAmount"), field)


def _fingerprint_payload(snapshot: NormalizedSnapshot) -> dict[str, Any]:
    return {
        "holdings": [asdict(item) for item in snapshot.holdings],
        "cash": [asdict(item) for item in snapshot.cash],
        "fx_rate": asdict(snapshot.fx_rate) if snapshot.fx_rate else None,
        "orders": [asdict(item) for item in snapshot.orders],
        "reconciliation": snapshot.reconciliation,
    }


def _fingerprint(snapshot: NormalizedSnapshot) -> str:
    payload = json.dumps(
        _fingerprint_payload(snapshot),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return sha256(payload).hexdigest()


class TossObservationService:
    """Collect and normalize one read-only Toss account observation."""

    def __init__(
        self,
        config: TossApiConfig,
        reader: TossAuthorizedReader,
        *,
        clock: Callable[[], float] = time.time,
        freshness_seconds: float = 300.0,
    ):
        self._config = config
        self._reader = reader
        self._clock = clock
        self._freshness_seconds = freshness_seconds

    def collect(
        self,
        *,
        from_date: str | None = None,
        to_date: str | None = None,
        max_order_pages: int = 100,
    ) -> NormalizedSnapshot:
        started_at = _now_iso(self._clock)
        quality: dict[str, Any] = {}
        reconciliation: dict[str, Any] = {}
        try:
            self._validate_account()
        except Exception as exc:
            return self._failed_snapshot(started_at, "accounts", exc)

        try:
            holdings, overview = self._read_holdings()
        except Exception as exc:
            return self._failed_snapshot(started_at, "holdings", exc)

        issues: list[str] = []
        try:
            cash = self._read_cash()
            quality["cash"] = "ok"
        except Exception as exc:
            cash = []
            quality["cash"] = {"error": str(exc)}
            issues.append("cash")

        try:
            fx_rate = self._read_fx_rate()
            quality["exchange_rate"] = "ok"
        except Exception as exc:
            fx_rate = None
            quality["exchange_rate"] = {"error": str(exc)}
            issues.append("exchange_rate")

        try:
            orders, order_quality = self._read_orders(
                from_date=from_date,
                to_date=to_date,
                max_order_pages=max_order_pages,
            )
            quality["orders"] = order_quality
            if order_quality.get("repeated_cursor"):
                issues.append("orders")
        except Exception as exc:
            orders = []
            quality["orders"] = {"error": str(exc)}
            issues.append("orders")

        holdings, holdings_reconciliation = self._apply_fx(holdings, fx_rate, overview)
        cash = self._apply_cash_fx(cash, fx_rate)
        reconciliation["holdings"] = holdings_reconciliation
        reconciliation["cash_currency_check"] = self._cash_currency_check(cash)
        if not holdings_reconciliation.get("all_within_tolerance", False):
            issues.append("holdings_reconciliation")
        if reconciliation["cash_currency_check"] != "verified_by_currency_labels":
            issues.append("cash_currency")
        if fx_rate is None and any(
            item.currency == "USD" for item in holdings + tuple(cash)
        ):
            issues.append("exchange_rate")

        state = SyncState.PARTIAL if issues else SyncState.COMPLETE
        if state == SyncState.COMPLETE and self._is_stale(fx_rate):
            state = SyncState.STALE
            quality["stale"] = True
        overview_invested_value_krw = self._overview_total_krw(overview, fx_rate)
        invested_value_krw = (
            overview_invested_value_krw
            if overview_invested_value_krw is not None
            else self._sum_optional(item.market_value_krw for item in holdings)
        )
        cash_value_krw = self._sum_optional(item.buying_power_krw for item in cash)
        total_value_krw = (
            invested_value_krw + cash_value_krw
            if invested_value_krw is not None and cash_value_krw is not None
            else None
        )
        snapshot = NormalizedSnapshot(
            account_alias=ACCOUNT_ALIAS,
            sync_started_at=started_at,
            synced_at=_now_iso(self._clock),
            state=state,
            holdings=tuple(holdings),
            cash=tuple(cash),
            fx_rate=fx_rate,
            orders=tuple(orders),
            source_timestamps={
                "exchange_rate_valid_from": fx_rate.valid_from if fx_rate else None,
                "exchange_rate_valid_until": fx_rate.valid_until if fx_rate else None,
            },
            data_quality={"issues": issues, **quality},
            reconciliation=reconciliation,
            total_value_krw=total_value_krw,
            invested_value_krw=invested_value_krw,
            cash_value_krw=cash_value_krw,
            fingerprint="",
        )
        return replace(snapshot, fingerprint=_fingerprint(snapshot))

    def sync(
        self,
        *,
        from_date: str | None = None,
        to_date: str | None = None,
        max_order_pages: int = 100,
    ) -> dict[str, Any]:
        """Collect and persist one immutable observation snapshot."""
        from storage.account_observation_store import insert_snapshot

        return insert_snapshot(
            self.collect(
                from_date=from_date,
                to_date=to_date,
                max_order_pages=max_order_pages,
            )
        )

    def health(self) -> dict[str, Any]:
        """Check configuration, OAuth, account discovery, and configured match."""
        try:
            payload = self._reader.get_json(
                "/api/v1/accounts", include_account_header=False
            )
            accounts = payload.get("result")
            if not isinstance(accounts, list):
                raise ObservationError("accounts result is not a list")
            brokerage_accounts = [
                account
                for account in accounts
                if isinstance(account, Mapping)
                and account.get("accountType") == "BROKERAGE"
            ]
            matched = any(
                int(account.get("accountSeq")) == self._config.account_seq
                for account in brokerage_accounts
                if str(account.get("accountSeq", "")).isdigit()
            )
            if not matched:
                raise ObservationError("configured brokerage account was not found")
            return {
                "ok": True,
                "checks": {
                    "config": "ok",
                    "oauth": "ok",
                    "account_discovery": "ok",
                    "account_match": "ok",
                },
                "account_count": len(brokerage_accounts),
                "error": None,
            }
        except Exception as exc:
            return {
                "ok": False,
                "checks": {
                    "config": "ok",
                    "oauth": "unknown",
                    "account_discovery": "failed",
                },
                "account_count": 0,
                "error": {"stage": "accounts", "message": str(exc)},
            }

    def _validate_account(self) -> None:
        payload = self._reader.get_json(
            "/api/v1/accounts", include_account_header=False
        )
        accounts = payload.get("result")
        if not isinstance(accounts, list):
            raise ObservationError("accounts result is not a list")
        for account in accounts:
            if not isinstance(account, Mapping):
                continue
            try:
                account_seq = int(account.get("accountSeq"))
            except (TypeError, ValueError):
                continue
            if (
                account_seq == self._config.account_seq
                and account.get("accountType") == "BROKERAGE"
            ):
                return
        raise ObservationError("configured brokerage account was not found")

    def _is_stale(self, fx_rate: NormalizedFxRate | None) -> bool:
        if fx_rate is None or not fx_rate.valid_until:
            return False
        try:
            valid_until = datetime.fromisoformat(
                fx_rate.valid_until.replace("Z", "+00:00")
            )
        except ValueError:
            return False
        now = datetime.fromtimestamp(self._clock(), timezone.utc)
        return (now - valid_until).total_seconds() > self._freshness_seconds

    def _read_holdings(self) -> tuple[list[NormalizedHolding], Mapping[str, Any]]:
        payload = self._reader.get_json("/api/v1/holdings")
        result = payload.get("result")
        if not isinstance(result, Mapping):
            raise ObservationError("holdings result is not an object")
        items = result.get("items")
        if not isinstance(items, list):
            raise ObservationError("holdings items is not a list")
        normalized = []
        for item in items:
            if not isinstance(item, Mapping):
                raise ObservationError("holding item is not an object")
            currency = str(item.get("currency") or "")
            if currency not in {"KRW", "USD"}:
                raise ObservationError("unsupported holding currency")
            normalized.append(
                NormalizedHolding(
                    symbol=str(item.get("symbol") or ""),
                    name=str(item.get("name") or ""),
                    market_country=str(item.get("marketCountry") or ""),
                    currency=currency,
                    quantity=float(
                        _decimal(item.get("quantity"), "quantity", nonnegative=True)
                    ),
                    last_price=float(
                        _decimal(item.get("lastPrice"), "lastPrice", nonnegative=True)
                    ),
                    average_purchase_price=float(
                        _decimal(
                            item.get("averagePurchasePrice"),
                            "averagePurchasePrice",
                            nonnegative=True,
                        )
                    ),
                    market_value_native=float(
                        _market_amount(item, "marketValue", "marketValue.amount")
                    ),
                    market_value_krw=None,
                    cost_native=float(_purchase_amount(item, "cost.purchaseAmount")),
                    cost_krw=None,
                    profit_loss_native=float(
                        _market_amount(item, "profitLoss", "profitLoss.amount")
                    ),
                    profit_loss_krw=None,
                    daily_profit_loss_native=float(
                        _market_amount(
                            item, "dailyProfitLoss", "dailyProfitLoss.amount"
                        )
                    ),
                    daily_profit_loss_krw=None,
                )
            )
        return normalized, result

    def _read_cash(self) -> list[NormalizedCash]:
        values = []
        for currency in ("KRW", "USD"):
            payload = self._reader.get_json(
                "/api/v1/buying-power", params={"currency": currency}
            )
            result = payload.get("result")
            if not isinstance(result, Mapping) or result.get("currency") != currency:
                raise ObservationError(f"buying power currency mismatch: {currency}")
            native = _decimal(
                result.get("cashBuyingPower"),
                f"{currency}.cashBuyingPower",
                nonnegative=True,
            )
            values.append(NormalizedCash(currency, float(native), None))
        return values

    def _read_fx_rate(self) -> NormalizedFxRate:
        payload = self._reader.get_json(
            "/api/v1/exchange-rate",
            params={"baseCurrency": "USD", "quoteCurrency": "KRW"},
        )
        result = payload.get("result")
        if not isinstance(result, Mapping):
            raise ObservationError("exchange rate result is not an object")
        if result.get("baseCurrency") != "USD" or result.get("quoteCurrency") != "KRW":
            raise ObservationError("exchange rate currency mismatch")
        rate = _decimal(result.get("rate"), "exchange rate", nonnegative=True)
        if rate <= 0:
            raise ObservationError("exchange rate must be positive")
        return NormalizedFxRate(
            base_currency="USD",
            quote_currency="KRW",
            rate=float(rate),
            mid_rate=_float(_optional_decimal(result.get("midRate"), "midRate")),
            valid_from=result.get("validFrom"),
            valid_until=result.get("validUntil"),
        )

    def _read_orders(
        self,
        *,
        from_date: str | None,
        to_date: str | None,
        max_order_pages: int,
    ) -> tuple[list[NormalizedOrder], dict[str, Any]]:
        if max_order_pages <= 0:
            raise ObservationError("max_order_pages must be positive")
        orders: list[NormalizedOrder] = []
        seen_ids: set[str] = set()
        seen_cursors: set[str] = set()
        cursor: str | None = None
        repeated_cursor: str | None = None
        pages = 0
        while pages < max_order_pages:
            params: dict[str, str | int] = {"status": "CLOSED", "limit": 100}
            if from_date:
                params["from"] = from_date
            if to_date:
                params["to"] = to_date
            if cursor:
                if cursor in seen_cursors:
                    repeated_cursor = cursor
                    break
                seen_cursors.add(cursor)
                params["cursor"] = cursor
            payload = self._reader.get_json("/api/v1/orders", params=params)
            result = payload.get("result")
            if not isinstance(result, Mapping) or not isinstance(
                result.get("orders"), list
            ):
                raise ObservationError("orders result is invalid")
            pages += 1
            for item in result["orders"]:
                normalized = self._normalize_order(item)
                if normalized.order_id not in seen_ids:
                    seen_ids.add(normalized.order_id)
                    orders.append(normalized)
            if not result.get("hasNext"):
                break
            next_cursor = result.get("nextCursor")
            if not isinstance(next_cursor, str) or not next_cursor:
                raise ObservationError("orders nextCursor is missing")
            if next_cursor in seen_cursors:
                repeated_cursor = next_cursor
                break
            cursor = next_cursor
        else:
            raise ObservationError("orders pagination exceeded max_order_pages")
        return orders, {
            "pages": pages,
            "count": len(orders),
            "repeated_cursor": repeated_cursor,
        }

    def _normalize_order(self, item: Any) -> NormalizedOrder:
        if not isinstance(item, Mapping):
            raise ObservationError("order item is not an object")
        execution = item.get("execution") or {}
        if not isinstance(execution, Mapping):
            raise ObservationError("order execution is not an object")
        return NormalizedOrder(
            order_id=str(item.get("orderId") or ""),
            symbol=str(item.get("symbol") or ""),
            currency=str(item.get("currency") or ""),
            side=str(item.get("side") or ""),
            order_type=str(item.get("orderType") or ""),
            status=str(item.get("status") or ""),
            ordered_at=item.get("orderedAt"),
            canceled_at=item.get("canceledAt"),
            quantity=float(
                _decimal(item.get("quantity"), "order.quantity", nonnegative=True)
            ),
            order_price_native=_float(
                _optional_decimal(item.get("price"), "order.price")
            ),
            order_amount_native=_float(
                _optional_decimal(item.get("orderAmount"), "order.orderAmount")
            ),
            filled_quantity=float(
                _decimal(
                    execution.get("filledQuantity", "0"),
                    "filledQuantity",
                    nonnegative=True,
                )
            ),
            average_filled_price_native=_float(
                _optional_decimal(
                    execution.get("averageFilledPrice"), "averageFilledPrice"
                )
            ),
            filled_amount_native=_float(
                _optional_decimal(execution.get("filledAmount"), "filledAmount")
            ),
            commission_native=_float(
                _optional_decimal(execution.get("commission"), "commission")
            ),
            tax_native=_float(_optional_decimal(execution.get("tax"), "tax")),
            filled_at=execution.get("filledAt"),
            settlement_date=execution.get("settlementDate"),
        )

    def _apply_fx(
        self,
        holdings: list[NormalizedHolding],
        fx_rate: NormalizedFxRate | None,
        overview: Mapping[str, Any],
    ) -> tuple[list[NormalizedHolding], dict[str, Any]]:
        rate = Decimal(str(fx_rate.rate)) if fx_rate else None
        by_currency: dict[str, Decimal] = {}
        for item in holdings:
            by_currency[item.currency] = by_currency.get(
                item.currency, Decimal("0")
            ) + Decimal(str(item.market_value_native))
        market_value = overview.get("marketValue")
        if not isinstance(market_value, Mapping):
            market_value = {}
        comparisons = {}
        all_within = True
        for currency, total in by_currency.items():
            overview_value = market_value.get(currency.lower())
            overview_total = _optional_decimal(
                overview_value, f"overview.marketValue.{currency.lower()}"
            )
            within_tolerance = (
                overview_total is not None and abs(total - overview_total) <= TOLERANCE
            )
            comparisons[currency] = {
                "item_total_native": float(total),
                "overview_total_native": _float(overview_total),
                "within_tolerance": within_tolerance,
            }
            all_within = all_within and within_tolerance
        updated = []
        for item in holdings:
            multiplier = Decimal("1") if item.currency == "KRW" else rate
            updated.append(
                replace(
                    item,
                    market_value_krw=_float(
                        Decimal(str(item.market_value_native)) * multiplier
                    )
                    if multiplier is not None
                    else None,
                    cost_krw=_float(Decimal(str(item.cost_native)) * multiplier)
                    if multiplier is not None
                    else None,
                    profit_loss_krw=_float(
                        Decimal(str(item.profit_loss_native)) * multiplier
                    )
                    if multiplier is not None
                    else None,
                    daily_profit_loss_krw=_float(
                        Decimal(str(item.daily_profit_loss_native)) * multiplier
                    )
                    if multiplier is not None
                    else None,
                )
            )
        return updated, {
            "by_currency": comparisons,
            **comparisons,
            "all_within_tolerance": all_within,
        }

    def _apply_cash_fx(
        self,
        cash: list[NormalizedCash],
        fx_rate: NormalizedFxRate | None,
    ) -> list[NormalizedCash]:
        rate = Decimal(str(fx_rate.rate)) if fx_rate else None
        updated = []
        for item in cash:
            multiplier = Decimal("1") if item.currency == "KRW" else rate
            updated.append(
                replace(
                    item,
                    buying_power_krw=_float(
                        Decimal(str(item.buying_power_native)) * multiplier
                    )
                    if multiplier is not None
                    else None,
                )
            )
        return updated

    def _cash_currency_check(self, cash: list[NormalizedCash]) -> str:
        currencies = {item.currency for item in cash}
        return (
            "verified_by_currency_labels"
            if currencies == {"KRW", "USD"}
            else "incomplete_currency_set"
        )

    def _overview_total_krw(
        self, overview: Mapping[str, Any], fx_rate: NormalizedFxRate | None
    ) -> float | None:
        market_value = overview.get("marketValue")
        if not isinstance(market_value, Mapping):
            return None
        krw = _optional_decimal(market_value.get("krw"), "overview.marketValue.krw")
        usd = _optional_decimal(market_value.get("usd"), "overview.marketValue.usd")
        if krw is None:
            return None
        if usd is not None and fx_rate is None:
            return None
        return float(krw + (usd * Decimal(str(fx_rate.rate)) if usd is not None else 0))

    def _sum_optional(self, values: Any) -> float | None:
        converted = [value for value in values if value is not None]
        return (
            float(sum(Decimal(str(value)) for value in converted))
            if converted
            else None
        )

    def _failed_snapshot(
        self, started_at: str, stage: str, error: Exception
    ) -> NormalizedSnapshot:
        snapshot = NormalizedSnapshot(
            account_alias=ACCOUNT_ALIAS,
            sync_started_at=started_at,
            synced_at=_now_iso(self._clock),
            state=SyncState.FAILED,
            holdings=(),
            cash=(),
            fx_rate=None,
            orders=(),
            source_timestamps={},
            data_quality={"failed_stage": stage, "error": str(error)},
            reconciliation={},
            total_value_krw=None,
            invested_value_krw=None,
            cash_value_krw=None,
            fingerprint="",
        )
        return replace(snapshot, fingerprint=_fingerprint(snapshot))
