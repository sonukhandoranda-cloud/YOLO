from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
import time

from src.footwear.detector import Detection


@dataclass(frozen=True)
class FootwearDecision:
    status: str
    label: str
    confidence: float
    message: str

    @property
    def can_mark_attendance(self) -> bool:
        return self.status in {"present", "present_with_flag"}


class FootwearPolicy:
    def __init__(
        self,
        allowed_labels: set[str] | None = None,
        flagged_labels: set[str] | None = None,
    ) -> None:
        self.allowed_labels = {label.lower() for label in (allowed_labels or {"shoe", "footwear"})}
        self.flagged_labels = {label.lower() for label in (flagged_labels or {"sandal"})}

    def decide(self, detections: list[Detection]) -> FootwearDecision:
        if not detections:
            return FootwearDecision(
                status="absent",
                label="",
                confidence=0.0,
                message="No approved footwear detected",
            )

        best = max(detections, key=lambda detection: detection.confidence)
        label = best.label.lower()
        if label in self.allowed_labels:
            return FootwearDecision(
                status="present",
                label=best.label,
                confidence=best.confidence,
                message=f"Approved footwear detected: {best.label}",
            )
        if label in self.flagged_labels:
            return FootwearDecision(
                status="present_with_flag",
                label=best.label,
                confidence=best.confidence,
                message=f"Flagged footwear detected: {best.label}",
            )
        return FootwearDecision(
            status="absent",
            label=best.label,
            confidence=best.confidence,
            message=f"Unapproved footwear detected: {best.label}",
        )


class FootwearDecisionWindow:
    def __init__(self, window_size: int = 6, required_votes: int = 2, hold_ms: float = 1500.0) -> None:
        self.window_size = window_size
        self.required_votes = required_votes
        self.hold_ms = hold_ms
        self._decisions: deque[FootwearDecision] = deque(maxlen=window_size)
        self._last_accepted: FootwearDecision | None = None
        self._last_accepted_time = 0.0

    def add(self, decision: FootwearDecision) -> FootwearDecision:
        now = time.perf_counter()
        self._decisions.append(decision)

        accepted = [item for item in self._decisions if item.can_mark_attendance and item.label]
        if accepted:
            counts = Counter(item.label.lower() for item in accepted)
            label, votes = counts.most_common(1)[0]
            if votes >= self.required_votes:
                best = max(
                    (item for item in accepted if item.label.lower() == label),
                    key=lambda item: item.confidence,
                )
                stable = FootwearDecision(
                    status=best.status,
                    label=best.label,
                    confidence=best.confidence,
                    message=f"Stable footwear detected: {best.label}",
                )
                self._last_accepted = stable
                self._last_accepted_time = now
                return stable

        if self._last_accepted and (now - self._last_accepted_time) * 1000 <= self.hold_ms:
            return FootwearDecision(
                status=self._last_accepted.status,
                label=self._last_accepted.label,
                confidence=self._last_accepted.confidence,
                message=f"Holding recent footwear: {self._last_accepted.label}",
            )

        return decision
