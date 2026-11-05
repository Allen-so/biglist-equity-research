import pandas as pd

from src.core.pipeline_io import (
    build_run_id,
    export_preflight_artifacts,
    format_preflight_summary,
    latest_completed_session,
    select_universe_symbols,
    subset_daily_data,
)


def test_latest_completed_session():
    dates = pd.bdate_range(
        "2026-01-01",
        periods=5,
    )

    frame = pd.DataFrame(
        {
            "close": [
                1,
                2,
                3,
                4,
                5,
            ],
        },
        index=dates,
    )

    assert (
        latest_completed_session(
            frame
        )
        == dates[-1]
    )


def test_select_universe_symbols():
    snapshot = pd.DataFrame(
        {
            "symbol": [
                "AAA",
                "BBB",
                "CCC",
            ],

            "universe_pass": [
                True,
                False,
                True,
            ],
        }
    )

    result = (
        select_universe_symbols(
            snapshot
        )
    )

    assert result == [
        "AAA",
        "CCC",
    ]


def test_subset_daily_data():
    frame = pd.DataFrame(
        {
            "close": [
                10.0
            ]
        },
        index=[
            pd.Timestamp(
                "2026-01-01"
            )
        ],
    )

    result = subset_daily_data(
        {
            "AAA": frame,
            "BBB": frame,
        },
        [
            "AAA",
        ],
    )

    assert set(
        result
    ) == {
        "AAA"
    }


def test_run_id():
    value = build_run_id(
        as_of=pd.Timestamp(
            "2026-10-05 20:15:30"
        )
    )

    assert (
        value
        == "20261005_201530"
    )


def test_export_preflight_artifacts(
    tmp_path,
):
    snapshot = pd.DataFrame(
        {
            "symbol": [
                "AAA",
                "BBB",
            ],

            "universe_pass": [
                True,
                False,
            ],
        }
    )

    result = (
        export_preflight_artifacts(
            universe_snapshot=snapshot,
            output_dir=tmp_path,
            run_id="TEST_RUN",
        )
    )

    assert (
        result[
            "latest_csv"
        ].exists()
    )

    assert (
        result[
            "historical_csv"
        ].exists()
    )

    assert (
        result[
            "latest_parquet"
        ].exists()
    )


def test_preflight_summary():
    output = (
        format_preflight_summary(
            directory_count=8000,
            static_pass_count=5000,
            downloaded_count=4900,
            universe_pass_count=2500,
            as_of=pd.Timestamp(
                "2026-10-05"
            ),
        )
    )

    assert (
        "BIG LIST ENGINE — DATA PREFLIGHT"
        in output
    )

    assert (
        "FINAL UNIVERSE PASS   2500"
        in output
    )

    assert (
        "STATUS                OK"
        in output
    )