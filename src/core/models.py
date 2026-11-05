from dataclasses import dataclass
from datetime import date
from enum import Enum


class Timeframe(str, Enum):
    WEEKLY = "1w"
    DAILY = "1d"
    HOURLY = "1h"


class Classification(str, Enum):
    READY = "READY"
    DEVELOPING = "DEVELOPING"
    WATCH = "WATCH"
    EXTENDED = "EXTENDED"
    REJECTED = "REJECTED"
    INCOMPLETE = "INCOMPLETE"


class GateStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    NOT_AVAILABLE = "NOT_AVAILABLE"


@dataclass(frozen=True)
class ScanContext:
    as_of_date: date
    model_version: str
    universe_name: str = "US_COMMON_STOCKS"