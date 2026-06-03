from __future__ import annotations

from dataclasses import dataclass

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
