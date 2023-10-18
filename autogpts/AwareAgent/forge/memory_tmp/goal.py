from typing import Any
from pydantic import Field

from forge.memory_tmp.goal_status import GoalStatus
from forge.helpers.parser.loggable_base_model import LoggableBaseModel


class Goal(LoggableBaseModel):
    description: str = Field("The description of the goal.")
    ability: str = Field(
        "The name of the ability (only name, without arguments) that should be used to achieve the goal should be one of the available capabilities, is very important that you verify that the goal can be achieved using this ability."
    )
    validation_condition: str = Field(
        "The condition that should be met to validate the goal. It should be used as a post-condition for current goal and pre-condition for the next goal. E.g: 'The file 'x' is created properly.'"
    )
    status: str = Field(
        "Should be one of the following: NOT_STARTED, IN_PROGRESS, SUCCEEDED, FAILED"
    )

    def get_description(self) -> str:
        """Get a human-readable description of the current goal"""

        return f"Description: {self.description}\nability: {self.ability}\nValidation condition: {self.validation_condition}\nStatus: {self.status}"

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
            "ability": self.ability,
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
            ability=data["ability"],
            validation_condition=data["validation_condition"],
            status=data["status"],
        )
