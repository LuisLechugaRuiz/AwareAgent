from pydantic import Field
from typing import Any
from forge.helpers.parser import LoggableBaseModel


class Thought(LoggableBaseModel):
    text: str = Field(
        description="A description of your decision-making process. This should detail the thought process that led to your final decision."
    )
    reasoning: str = Field(
        description="The logical basis for your decision. This should include the considerations, principles, or facts that support your choice. "
    )
    criticism: str = Field(
        description="Constructive self-criticism of your decision. This field should never be empty as it encourages self-reflection and continuous improvement. It helps to identify areas of improvement in your reasoning."
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "reasoning": self.reasoning,
            "criticism": self.criticism,
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            text=data["text"],
            reasoning=data["reasoning"],
            criticism=data["criticism"],
        )

    def get_description(self) -> str:
        """Return a human-readable description of the thought."""

        return f"Thought: {self.text}\nReasoning: {self.reasoning}\nCriticism: {self.criticism}"
