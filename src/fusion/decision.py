from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass


@dataclass(frozen=True)
class StableIdentity:
    student_id: int
    count: int


class IdentityVoteWindow:
    def __init__(self, window_size: int, required_votes: int) -> None:
        self.window: deque[int] = deque(maxlen=window_size)
        self.required_votes = required_votes

    def add(self, student_id: int | None) -> StableIdentity | None:
        if student_id is not None:
            self.window.append(student_id)
        if not self.window:
            return None
        student_id, count = Counter(self.window).most_common(1)[0]
        if count >= self.required_votes:
            return StableIdentity(student_id=student_id, count=count)
        return None
