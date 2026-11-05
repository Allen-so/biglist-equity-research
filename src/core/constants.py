MODEL_NAME = "Big List Engine"
MODEL_VERSION = "1.0.0"

REQUIRED_OHLCV_COLUMNS = (
    "open",
    "high",
    "low",
    "close",
    "volume",
)

DAILY_RETURN_WINDOWS = (
    1,
    5,
    10,
    20,
    60,
    120,
)

DAILY_MA_WINDOWS = (
    10,
    20,
    50,
    200,
)

WEEKLY_MA_WINDOWS = (
    10,
    20,
    40,
)

MARKET_INDEXES = (
    "SPY",
    "QQQ",
    "IWM",
)