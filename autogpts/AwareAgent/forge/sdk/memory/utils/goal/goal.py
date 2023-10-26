from typing import Any
from pydantic import Field

from forge.sdk.memory.utils.goal.goal_status import GoalStatus
from forge.helpers.parser.loggable_base_model import LoggableBaseModel


class Goal(LoggableBaseModel):
    description: str = Field("The description of the goal.")
    validation_condition: str = Field(
        "Explicit criteria acting as the benchmark for goal completion, essential for assessing the outcome's alignment with desired objectives. It serves as a conclusive checkpoint for the current goal and a foundational prerequisite for subsequent objectives"
    )
    status: str = Field(
        "Should be one of the following: NOT_STARTED, IN_PROGRESS, SUCCEEDED, FAILED"
    )

    def get_description(self) -> str:
        """Get a human-readable description of the current goal"""

        return f"Description: {self.description}\nValidation condition: {self.validation_condition}\nStatus: {self.status}"

    def get_status(self) -> GoalStatus:
        """Get the status of the goal"""

        return GoalStatus[self.status]

    def update_status(self, status: GoalStatus) -> None:
        """Update the status of the goal"""

        self.status = status.value

    def to_dict(self) -> dict[str, Any]:
        """Method used to serialize the goal"""

        return {
            "description": self.description,
            "validation_condition": self.validation_condition,
            "status": self.status,
        }

    def finished(self) -> bool:
        """Check if the goal has finished."""

        return self.get_status().finished()

    @classmethod
    def from_dict(cls, data):
        return cls(
            description=data["description"],
            validation_condition=data["validation_condition"],
            status=data["status"],
        )
