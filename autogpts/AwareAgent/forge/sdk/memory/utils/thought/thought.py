from pydantic import Field
from typing import Any
from forge.helpers.parser import LoggableBaseModel


class Thought(LoggableBaseModel):
    reasoning: str = Field(
        description="Explanation for your decision, encompassing supporting logic and areas for improvement. "
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "reasoning": self.reasoning,
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            reasoning=data["reasoning"],
        )

    def get_description(self) -> str:
        """Return a human-readable description of the thought."""

        return self.reasoning
