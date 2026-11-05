import numpy as np
import pandas as pd

from src.core.finalize import (
    build_final_lists,
    classify_final_candidate,
    dynamic_capacity,
    regime_from_score,
)
from src.output.formatter import (
    format_biglist_terminal,
)


def make_candidate(
    *,
    symbol: str,
    tradability: str = "READY",
    hourly: str = "NEUTRAL",
    admitted: bool = True,
    scoring_complete: bool = True,
    score_gate_pass: bool = True,
    total_score: float = 90.0,
) -> dict:
    return {
        "symbol":
            symbol,

        "admitted":
            admitted,

        "scoring_complete":
            scoring_complete,

        "score_gate_pass":
            score_gate_pass,

        "total_score":
            total_score,

        "normalized_available_score":
            total_score,

        "tradability_classification":
            tradability,

        "hourly_confirmation":
            hourly,
    }


def test_regime_from_score():
    assert (
        regime_from_score(
            80.0
        )
        == "STRONG_RISK_ON"
    )

    assert (
        regime_from_score(
            65.0
        )
        == "CONSTRUCTIVE"
    )

    assert (
        regime_from_score(
            50.0
        )
        == "SELECTIVE"
    )

    assert (
        regime_from_score(
            30.0
        )
        == "DEFENSIVE"
    )


def test_dynamic_capacity_boundaries():
    assert (
        dynamic_capacity(
            regime="STRONG_RISK_ON",
            regime_score=75.0,
        )
        == 45
    )

    assert (
        dynamic_capacity(
            regime="STRONG_RISK_ON",
            regime_score=100.0,
        )
        == 70
    )

    assert (
        dynamic_capacity(
            regime="CONSTRUCTIVE",
            regime_score=60.0,
        )
        == 35
    )

    assert (
        dynamic_capacity(
            regime="DEFENSIVE",
            regime_score=0.0,
        )
        == 10
    )


def test_ready_stays_ready_with_neutral_hourly():
    result = classify_final_candidate(
        make_candidate(
            symbol="AAA",
            tradability="READY",
            hourly="NEUTRAL",
        )
    )

    assert (
        result[
            "final_classification"
        ]
        == "READY"
    )


def test_negative_hourly_downgrades_ready_one_level():
    result = classify_final_candidate(
        make_candidate(
            symbol="AAA",
            tradability="READY",
            hourly="NEGATIVE",
        )
    )

    assert (
        result[
            "final_classification"
        ]
        == "DEVELOPING"
    )


def test_negative_hourly_downgrades_developing_to_watch():
    result = classify_final_candidate(
        make_candidate(
            symbol="AAA",
            tradability="DEVELOPING",
            hourly="NEGATIVE",
        )
    )

    assert (
        result[
            "final_classification"
        ]
        == "WATCH"
    )


def test_hard_gate_cannot_be_rescued():
    result = classify_final_candidate(
        make_candidate(
            symbol="AAA",
            tradability="READY",
            admitted=False,
            total_score=99.0,
        )
    )

    assert (
        result[
            "final_classification"
        ]
        == "REJECTED"
    )


def test_incomplete_score_is_not_called_rejected():
    result = classify_final_candidate(
        make_candidate(
            symbol="AAA",
            tradability="READY",
            scoring_complete=False,
        )
    )

    assert (
        result[
            "final_classification"
        ]
        == "INCOMPLETE"
    )


def test_capacity_is_ceiling_not_quota():
    candidates = pd.DataFrame(
        [
            make_candidate(
                symbol="AAA",
                total_score=95.0,
            ),
            make_candidate(
                symbol="BBB",
                total_score=92.0,
            ),
            make_candidate(
                symbol="CCC",
                total_score=90.0,
            ),
        ]
    )

    result = build_final_lists(
        candidates,
        regime="STRONG_RISK_ON",
        regime_score=80.0,
    )

    #
    # Capacity is much larger than 3,
    # but engine must NOT invent additional stocks.
    #
    assert (
        len(
            result[
                "biglist"
            ]
        )
        == 3
    )

    assert (
        result[
            "capacity_used"
        ]
        == 3
    )


def test_watch_is_not_part_of_biglist():
    candidates = pd.DataFrame(
        [
            make_candidate(
                symbol="AAA",
                tradability="READY",
            ),
            make_candidate(
                symbol="BBB",
                tradability="DEVELOPING",
            ),
            make_candidate(
                symbol="CCC",
                tradability="WATCH",
            ),
        ]
    )

    result = build_final_lists(
        candidates,
        regime="CONSTRUCTIVE",
        regime_score=65.0,
    )

    symbols = set(
        result[
            "biglist"
        ][
            "symbol"
        ]
    )

    assert symbols == {
        "AAA",
        "BBB",
    }

    assert (
        result[
            "watch"
        ][
            "symbol"
        ].tolist()
        == [
            "CCC"
        ]
    )


def test_terminal_output_format():
    candidates = pd.DataFrame(
        [
            make_candidate(
                symbol="AAA",
                tradability="READY",
                total_score=95.0,
            ),
            make_candidate(
                symbol="BBB",
                tradability="DEVELOPING",
                total_score=90.0,
            ),
            make_candidate(
                symbol="CCC",
                tradability="WATCH",
                total_score=85.0,
            ),
        ]
    )

    result = build_final_lists(
        candidates,
        regime="CONSTRUCTIVE",
        regime_score=65.0,
    )

    output = (
        format_biglist_terminal(
            result
        )
    )

    assert (
        "BIG LIST — 2"
        in output
    )

    assert (
        "READY — 1"
        in output
    )

    assert (
        "DEVELOPING — 1"
        in output
    )

    assert (
        "WATCH LIST — 1"
        in output
    )

    assert "AAA" in output
    assert "BBB" in output
    assert "CCC" in output


# Exercise the production admission -> finalization boundary, not a manually
# granted eligibility flag.
import pytest
from src.core.admission import evaluate_hard_gates


def make_setup_candidate(setup=4, *, channel=True, broken=False, trade=True,
                         tradability="READY", hourly="NEUTRAL", **kwargs):
    row = make_candidate(symbol="FORM", tradability=tradability, hourly=hourly, **kwargs)
    row["setup_score"] = setup
    row.update(evaluate_hard_gates(
        admission={"admission_channel_pass": channel},
        structure={"broken_structure_core": broken},
        setup_row=row,
        tradability_row={"tradability_gate_pass": trade},
    ))
    return row


@pytest.mark.parametrize("tradability", ["READY", "DEVELOPING", "WATCH"])
@pytest.mark.parametrize("hourly", ["SUPPORTIVE", "NEUTRAL", "NEGATIVE"])
def test_forming_is_always_watch_only(tradability, hourly):
    row = make_setup_candidate(tradability=tradability, hourly=hourly)
    assert not row["admitted"]
    assert classify_final_candidate(row) == {
        "final_classification": "WATCH",
        "classification_reason": "FORMING_SETUP_WATCH",
    }


@pytest.mark.parametrize("setup", [0, 1, 2, 3])
@pytest.mark.parametrize("tradability", ["READY", "EXTENDED"])
def test_low_setup_remains_rejected(setup, tradability):
    row = make_setup_candidate(setup, tradability=tradability)
    assert classify_final_candidate(row)["final_classification"] == "REJECTED"


@pytest.mark.parametrize("failure", [{"channel": False}, {"broken": True}, {"trade": False}])
@pytest.mark.parametrize("tradability", ["READY", "EXTENDED"])
def test_forming_cannot_rescue_true_hard_fail(failure, tradability):
    row = make_setup_candidate(tradability=tradability, **failure)
    assert not row["forming_watch_eligible"]
    assert classify_final_candidate(row)["final_classification"] == "REJECTED"


@pytest.mark.parametrize("overrides,reason", [
    ({"scoring_complete": False}, "SCORE_INCOMPLETE"),
    ({"score_gate_pass": False, "total_score": 79.9}, "REGIME_SCORE_FAIL"),
])
def test_forming_requires_complete_passing_score(overrides, reason):
    result = classify_final_candidate(make_setup_candidate(**overrides))
    assert result == {"final_classification": "REJECTED", "classification_reason": reason}


@pytest.mark.parametrize("tradability", ["EXTENDED", "REJECTED", "INCOMPLETE", "UNKNOWN"])
def test_forming_invalid_location_cannot_be_watch(tradability):
    result = classify_final_candidate(make_setup_candidate(tradability=tradability))
    assert result["final_classification"] == "REJECTED"


@pytest.mark.parametrize("setup", [5, 8])
@pytest.mark.parametrize("tradability", ["READY", "DEVELOPING", "WATCH", "EXTENDED"])
@pytest.mark.parametrize("hourly", ["SUPPORTIVE", "NEGATIVE"])
def test_formal_admission_classification_is_unchanged(setup, tradability, hourly):
    row = make_setup_candidate(setup, tradability=tradability, hourly=hourly)
    assert row["admitted"]
    assert not row["forming_watch_eligible"]
    assert classify_final_candidate(row) == classify_final_candidate(
        make_candidate(symbol="FORM", tradability=tradability, hourly=hourly)
    )


@pytest.mark.parametrize("eligible", [None, np.nan, False])
def test_missing_forming_eligibility_fails_closed(eligible):
    row = make_setup_candidate()
    row["forming_watch_eligible"] = eligible
    assert classify_final_candidate(row)["final_classification"] == "REJECTED"


def test_forming_watch_does_not_fill_biglist_capacity():
    rows = [make_setup_candidate(), make_candidate(symbol="FORMAL")]
    result = build_final_lists(pd.DataFrame(rows), regime="SELECTIVE", regime_score=50)
    assert result["biglist"].symbol.tolist() == ["FORMAL"]
    assert result["watch"].symbol.tolist() == ["FORM"]
    assert result["capacity_used"] == 1 < result["capacity"]
    assert not result["watch"].selected_biglist.any()


@pytest.mark.parametrize("complete,expected", [(True, "COMPLETE"), (False, "INCOMPLETE")])
def test_data_status_is_independent_of_hard_gate_rejection(complete, expected):
    row = make_candidate(symbol="DPC", admitted=False, scoring_complete=complete)
    row["rejection_reasons"] = ["STRUCTURE_FAIL", "TRADABILITY_FAIL"]
    result = build_final_lists(pd.DataFrame([row]), regime="SELECTIVE")["full"].iloc[0]
    assert result.data_status == expected
    assert result.final_classification == "REJECTED"
    assert result.classification_reason == "HARD_GATE_FAIL"
    assert result.rejection_reasons == row["rejection_reasons"]


def test_data_status_does_not_change_selection(monkeypatch):
    import src.core.finalize as finalize
    rows = [make_candidate(symbol=f"READY{i}", total_score=99-i) for i in range(40)]
    rows += [make_setup_candidate(), make_candidate(symbol="MISSING", scoring_complete=False),
             make_candidate(symbol="FAIL", admitted=False, scoring_complete=False)]
    candidates = pd.DataFrame(rows)
    original = finalize.apply_final_classification
    actual = build_final_lists(candidates, regime="SELECTIVE", regime_score=50)
    def without_data_status(frame):
        return original(frame).drop(columns="data_status")
    monkeypatch.setattr(finalize, "apply_final_classification", without_data_status)
    expected = build_final_lists(candidates, regime="SELECTIVE", regime_score=50)
    for key, value in expected.items():
        if isinstance(value, pd.DataFrame):
            pd.testing.assert_frame_equal(actual[key].drop(columns="data_status"), value)
        else:
            assert actual[key] == value
    assert actual["capacity_used"] == actual["capacity"]
    assert actual["watch"].symbol.tolist() == ["FORM"]
