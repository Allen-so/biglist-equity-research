import numpy as np
import pandas as pd

from src.core.scoring import (
    build_score_snapshot,
    regime_minimum_score,
    score_candidate,
    score_progress_retention,
    score_rank_level_delta,
    score_volatility,
    validate_factor_weights,
)
from src.features.quality_factors import (
    build_quality_factor_snapshot,
)


def make_complete_candidate():
    return {
        "symbol": "AAA",

        "admitted": True,

        "market_fit_pct": 90.0,
        "group_fit_pct": 85.0,
        "structure_score": 7.0,
        "monster_quality_pct": 90.0,

        "volatility_pct": 85.0,
        "atr_pct": 0.05,

        "historical_velocity_pct": 92.0,
        "productive_volatility_pct": 88.0,
        "supply_demand_pct": 90.0,

        "progress_retention_10": 0.75,

        "persistent_rs": 93.0,

        "rs_20d_pct": 95.0,
        "rs_60d_pct": 90.0,

        "rs_20d_rank_delta": 20.0,
        "rs_60d_rank_delta": 10.0,

        "within_group_leadership_pct": 95.0,

        "fundamental_delta_status": "VALUE",
        "fundamental_delta_pct": 80.0,

        "setup_score": 8.0,
        "tradability_score": 4.5,
    }


def test_factor_weights_total_100():
    validate_factor_weights()


def test_progress_retention_bands():
    assert (
        score_progress_retention(
            0.20
        )
        == 0.0
    )

    assert (
        score_progress_retention(
            0.40
        )
        == 2.0
    )

    assert (
        score_progress_retention(
            0.55
        )
        == 4.0
    )

    assert (
        score_progress_retention(
            0.70
        )
        == 5.5
    )

    assert (
        score_progress_retention(
            0.85
        )
        == 7.0
    )


def test_volatility_prefers_three_to_eight_percent_atr():
    ideal = score_volatility(
        volatility_pct=100.0,
        atr_pct=0.05,
    )

    slow = score_volatility(
        volatility_pct=100.0,
        atr_pct=0.015,
    )

    extreme = score_volatility(
        volatility_pct=100.0,
        atr_pct=0.15,
    )

    assert np.isclose(
        ideal,
        7.0,
    )

    assert ideal > slow
    assert ideal > extreme


def test_rank_level_delta():
    result = (
        score_rank_level_delta(
            rs20=90.0,
            rs60=80.0,
            rs20_delta=20.0,
            rs60_delta=20.0,
        )
    )

    expected_level = (
        0.40 * 90.0
        + 0.60 * 80.0
    )

    assert np.isclose(
        result[
            "rank_level_pct"
        ],
        expected_level,
    )

    assert np.isclose(
        result[
            "rank_delta_points"
        ],
        2.0,
    )

    assert (
        result[
            "rank_level_delta_points"
        ]
        <= 6.0
    )


def test_complete_candidate_gets_total_score():
    result = score_candidate(
        make_complete_candidate()
    )

    assert result[
        "scoring_complete"
    ]

    assert (
        result[
            "score_status"
        ]
        == "COMPLETE"
    )

    assert not pd.isna(
        result[
            "total_score"
        ]
    )

    assert (
        0
        <= result[
            "total_score"
        ]
        <= 100
    )


def test_missing_fundamentals_are_not_imputed():
    candidate = (
        make_complete_candidate()
    )

    candidate[
        "fundamental_delta_status"
    ] = "MISSING"

    candidate[
        "fundamental_delta_pct"
    ] = np.nan

    result = score_candidate(
        candidate
    )

    # Fundamental data remains genuinely missing:
    # it is not silently imputed as zero.
    assert pd.isna(
        result[
            "fundamental_delta_points"
        ]
    )

    assert (
        "fundamental_delta"
        in result[
            "missing_factors"
        ]
    )

    # But fundamental_delta is optional data coverage,
    # so the remaining model can still produce a valid
    # normalized 0-100 score.
    assert result[
        "scoring_complete"
    ]

    assert (
        result[
            "score_status"
        ]
        == "COMPLETE"
    )

    assert (
        result[
            "available_maximum_points"
        ]
        == 95.0
    )

    assert (
        result[
            "total_score"
        ]
        == result[
            "normalized_available_score"
        ]
    )


def test_missing_core_factor_remains_incomplete():
    candidate = (
        make_complete_candidate()
    )

    candidate[
        "market_fit_pct"
    ] = np.nan

    result = score_candidate(
        candidate
    )

    assert not result[
        "scoring_complete"
    ]

    assert (
        result[
            "score_status"
        ]
        == "INCOMPLETE"
    )

    assert pd.isna(
        result[
            "total_score"
        ]
    )

    assert (
        "market_fit"
        in result[
            "missing_factors"
        ]
    )


def test_multiple_optional_factors_can_be_normalized():
    candidate = (
        make_complete_candidate()
    )

    candidate[
        "group_fit_pct"
    ] = np.nan

    candidate[
        "within_group_leadership_pct"
    ] = np.nan

    candidate[
        "fundamental_delta_status"
    ] = "MISSING"

    candidate[
        "fundamental_delta_pct"
    ] = np.nan

    result = score_candidate(
        candidate
    )

    assert result[
        "scoring_complete"
    ]

    assert (
        result[
            "available_maximum_points"
        ]
        == 87.0
    )

    assert set(
        result[
            "missing_factors"
        ]
    ) == {
        "group_fit",
        "group_leadership",
        "fundamental_delta",
    }

    assert (
        result[
            "total_score"
        ]
        == result[
            "normalized_available_score"
        ]
    )


def test_regime_thresholds():
    assert (
        regime_minimum_score(
            "strong risk on"
        )
        == 75.0
    )

    assert (
        regime_minimum_score(
            "constructive"
        )
        == 77.0
    )

    assert (
        regime_minimum_score(
            "selective"
        )
        == 80.0
    )

    assert (
        regime_minimum_score(
            "defensive"
        )
        == 83.0
    )


def test_hard_gate_cannot_be_overridden_by_score():
    strong = (
        make_complete_candidate()
    )

    strong[
        "admitted"
    ] = False

    result = build_score_snapshot(
        pd.DataFrame(
            [
                strong
            ]
        ),
        regime="CONSTRUCTIVE",
    )

    assert not bool(
        result[
            "final_score_eligible"
        ].iloc[0]
    )


def test_quality_factor_snapshot():
    snapshot = pd.DataFrame(
        {
            "symbol": [
                "FAST",
                "MEDIUM",
                "SLOW",
            ],

            "p90_return_5d": [
                0.20,
                0.10,
                0.04,
            ],

            "p90_return_10d": [
                0.30,
                0.18,
                0.07,
            ],

            "p90_return_20d": [
                0.45,
                0.25,
                0.10,
            ],

            "impulse_frequency": [
                0.30,
                0.15,
                0.05,
            ],

            "progress_retention_10": [
                0.90,
                0.60,
                0.30,
            ],

            "atr_pct": [
                0.06,
                0.04,
                0.02,
            ],

            "range_pct_20": [
                0.07,
                0.05,
                0.025,
            ],

            "up_down_volume_ratio_20": [
                2.5,
                1.2,
                0.7,
            ],

            "volume_weighted_clv_20": [
                0.60,
                0.10,
                -0.30,
            ],

            "accumulation_distribution_balance_20": [
                0.30,
                0.05,
                -0.20,
            ],
        }
    )

    result = (
        build_quality_factor_snapshot(
            snapshot
        )
        .set_index(
            "symbol"
        )
    )

    assert (
        result.loc[
            "FAST",
            "monster_quality_pct",
        ]
        >
        result.loc[
            "MEDIUM",
            "monster_quality_pct",
        ]
        >
        result.loc[
            "SLOW",
            "monster_quality_pct",
        ]
    )

    assert (
        result.loc[
            "FAST",
            "supply_demand_pct",
        ]
        >
        result.loc[
            "SLOW",
            "supply_demand_pct",
        ]
    )


def test_score_snapshot_passes_regime_gate():
    candidate = (
        make_complete_candidate()
    )

    candidate.update(
        {
            "market_fit_pct": 100.0,
            "group_fit_pct": 100.0,
            "structure_score": 8.0,
            "monster_quality_pct": 100.0,

            "volatility_pct": 100.0,
            "atr_pct": 0.05,

            "historical_velocity_pct": 100.0,
            "productive_volatility_pct": 100.0,
            "supply_demand_pct": 100.0,

            "progress_retention_10": 0.90,

            "persistent_rs": 100.0,

            "rs_20d_pct": 100.0,
            "rs_60d_pct": 100.0,

            "rs_20d_rank_delta": 20.0,
            "rs_60d_rank_delta": 20.0,

            "within_group_leadership_pct": 100.0,

            "fundamental_delta_status": "VALUE",
            "fundamental_delta_pct": 100.0,

            "setup_score": 10.0,
            "tradability_score": 5.0,
        }
    )

    result = build_score_snapshot(
        pd.DataFrame(
            [
                candidate
            ]
        ),
        regime="DEFENSIVE",
    )

    assert bool(
        result[
            "score_gate_pass"
        ].iloc[0]
    )

    assert bool(
        result[
            "final_score_eligible"
        ].iloc[0]
    )

    assert np.isclose(
        result[
            "total_score"
        ].iloc[0],
        100.0,
    )