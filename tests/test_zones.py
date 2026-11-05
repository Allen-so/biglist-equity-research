import pandas as pd

from src.features.zones import (
    build_zones,
    classify_zone_role,
    nearest_zone,
)


def make_swings() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "pivot_time":
                    pd.Timestamp("2026-01-05"),
                "confirmed_at":
                    pd.Timestamp("2026-01-07"),
                "kind": "HIGH",
                "price": 100.10,
                "atr": 2.0,
                "prominence_atr": 1.5,
            },
            {
                "pivot_time":
                    pd.Timestamp("2026-01-15"),
                "confirmed_at":
                    pd.Timestamp("2026-01-19"),
                "kind": "LOW",
                "price": 99.70,
                "atr": 2.1,
                "prominence_atr": 1.2,
            },
            {
                "pivot_time":
                    pd.Timestamp("2026-02-02"),
                "confirmed_at":
                    pd.Timestamp("2026-02-04"),
                "kind": "HIGH",
                "price": 100.40,
                "atr": 2.0,
                "prominence_atr": 1.8,
            },
            {
                "pivot_time":
                    pd.Timestamp("2026-02-18"),
                "confirmed_at":
                    pd.Timestamp("2026-02-20"),
                "kind": "LOW",
                "price": 99.90,
                "atr": 1.9,
                "prominence_atr": 1.4,
            },

            # Distant unrelated pivots
            {
                "pivot_time":
                    pd.Timestamp("2026-01-10"),
                "confirmed_at":
                    pd.Timestamp("2026-01-12"),
                "kind": "HIGH",
                "price": 112.0,
                "atr": 2.0,
                "prominence_atr": 1.5,
            },
            {
                "pivot_time":
                    pd.Timestamp("2026-02-10"),
                "confirmed_at":
                    pd.Timestamp("2026-02-12"),
                "kind": "HIGH",
                "price": 128.0,
                "atr": 2.0,
                "prominence_atr": 1.5,
            },

            # Future pivot:
            # must not exist for an earlier as-of date
            {
                "pivot_time":
                    pd.Timestamp("2026-04-01"),
                "confirmed_at":
                    pd.Timestamp("2026-04-03"),
                "kind": "LOW",
                "price": 100.0,
                "atr": 2.0,
                "prominence_atr": 2.0,
            },
        ]
    )


def test_builds_role_reversal_zone():
    swings = make_swings()

    zones = build_zones(
        swings,
        as_of=pd.Timestamp("2026-03-01"),
        current_price=105.0,
        current_atr=2.0,
    )

    assert len(zones) == 1

    zone = zones.iloc[0]

    assert zone["pivot_count"] == 4
    assert zone["high_pivot_count"] == 2
    assert zone["low_pivot_count"] == 2

    assert zone["zone_low"] == 99.70
    assert zone["zone_high"] == 100.40

    assert zone["role"] == "SUPPORT"


def test_distant_pivots_do_not_chain_into_zone():
    swings = make_swings()

    zones = build_zones(
        swings,
        as_of=pd.Timestamp("2026-03-01"),
        current_price=105.0,
        current_atr=2.0,
    )

    assert len(zones) == 1

    zone = zones.iloc[0]

    assert zone["zone_high"] < 112.0


def test_future_pivot_is_not_used():
    swings = make_swings()

    zones_before = build_zones(
        swings,
        as_of=pd.Timestamp("2026-03-01"),
        current_price=105.0,
        current_atr=2.0,
    )

    zone_before = zones_before.iloc[0]

    assert zone_before["pivot_count"] == 4

    zones_after = build_zones(
        swings,
        as_of=pd.Timestamp("2026-04-10"),
        current_price=105.0,
        current_atr=2.0,
    )

    zone_after = zones_after.iloc[0]

    assert zone_after["pivot_count"] == 5


def test_zone_role_classification():
    assert (
        classify_zone_role(
            zone_low=99.0,
            zone_high=101.0,
            current_price=105.0,
        )
        == "SUPPORT"
    )

    assert (
        classify_zone_role(
            zone_low=99.0,
            zone_high=101.0,
            current_price=95.0,
        )
        == "RESISTANCE"
    )

    assert (
        classify_zone_role(
            zone_low=99.0,
            zone_high=101.0,
            current_price=100.0,
        )
        == "CURRENT"
    )


def test_nearest_support_zone():
    swings = make_swings()

    zones = build_zones(
        swings,
        as_of=pd.Timestamp("2026-03-01"),
        current_price=105.0,
        current_atr=2.0,
    )

    support = nearest_zone(
        zones,
        role="SUPPORT",
    )

    assert support is not None

    assert (
        support["zone_low"]
        == 99.70
    )