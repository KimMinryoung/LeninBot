"""Stage result and usage records shared by the CommuLingo agent sessions.

The queue engine itself moved to the frontend (services/commulingo-pipeline).
"""
from dataclasses import dataclass, field


@dataclass
class Result:
    value: dict
    next_stage: str
    status: str = 'ready'
    delay_seconds: int = 0


@dataclass
class Usage:
    tracker: dict = field(default_factory=dict)
    prepared: dict = field(default_factory=dict)
    # Unknown usage after a process/provider failure retains the reservation.
    complete: bool = False
    started: bool = False
