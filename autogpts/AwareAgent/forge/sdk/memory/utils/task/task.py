from typing import Any, Dict

from forge.sdk.memory.utils.task.task_status import TaskStatus


class Task:
    def __init__(self, id: str, title: str, description: str, status: str):
        super().__init__()
        self.id = id
        self.title = title
        self.description = description
        self.status = TaskStatus.from_string(status)

    def get_description(self) -> str:
        """Get a human-readable description of the current goal"""

        return f"Title: {self.title}\nDescription: {self.description}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "status": self.status.value,
        }

    @classmethod
    def from_dict(cls, data):
        step = cls(
            id=data["id"],
            title=data["title"],
            description=data["description"],
            status=data["status"],
        )
        return step

    def __str__(self) -> str:
        return self.get_description()
