from enum import Enum


class GoalStatus(Enum):
    NOT_STARTED = "NOT_STARTED"
    IN_PROGRESS = "IN_PROGRESS"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"

    def finished(self) -> bool:
        """Check if the goal has finished."""

        return self == GoalStatus.SUCCEEDED or self == GoalStatus.FAILED
