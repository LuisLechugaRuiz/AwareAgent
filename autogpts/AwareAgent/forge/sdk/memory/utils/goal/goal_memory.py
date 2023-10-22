from typing import Any, List

from forge.sdk.memory.utils.goal.goal import Goal
from forge.sdk.config.config import Config


class GoalMemory:
    def __init__(
        self,
        max_iterations=Config().max_goal_iterations,
    ):
        self.goals: List[Goal] = []
        self.max_iterations = max_iterations
        self.iterations = 0

    def get_goals(self) -> List[Goal]:
        """Get goals"""

        return self.goals

    def max_iterations_reached(self) -> bool:
        """Check if the maximum number of iterations has been reached."""

        self.iterations += 1
        return self.iterations > self.max_iterations

    def set_goals(self, goals: List[Goal]) -> None:
        """Set goals"""

        self.goals = goals

    def to_dict(self) -> dict[str, Any]:
        """Goal memory to dict"""

        return {
            "goals": [goal.to_dict() for goal in self.goals],
            "max_iterations": self.max_iterations,
            "iterations": self.iterations,
        }

    @classmethod
    def from_dict(cls, data):
        goal_memory = cls(
            max_iterations=data.get("max_iterations"),
        )
        for goal in data["goals"]:
            goal_memory.goals.append(Goal.from_dict(data=goal))
        goal_memory.iterations = data["iterations"]
        return goal_memory
