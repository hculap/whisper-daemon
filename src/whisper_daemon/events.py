"""Event types for inter-module communication."""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any


class EventType(Enum):
    RECORD_TOGGLE = auto()
    RECORD_STOP = auto()
    TRANSCRIPTION_DONE = auto()
    PASTE_LAST = auto()
    ERROR = auto()


@dataclass(frozen=True)
class Event:
    type: EventType
    payload: Any = None
