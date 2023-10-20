from datetime import datetime
from pydantic import Field
from typing import Tuple

from forge.memory_tmp import Goal, Thought
from forge.helpers.parser import ChatParser, LoggableBaseModel, get_json_schema
from forge.sdk import ForgeLogger, PromptEngine

LOG = ForgeLogger(__name__)


class Plan(LoggableBaseModel):
    goals: list[Goal] = Field(
        description="The updated list of goals that you should work on, consider previous goals and update them based on new information"
    )
    search_queries: list[str] = Field(
        description="A list of queries that need to be answered in order to retrieve the relevant context for the execution of the action tied to the highest priority goal"
    )

    @classmethod
    async def get_plan(
        cls,
        model,
        task,
        previous_thought,
        abilities,
        directories,
        summary=None,
        goals=None,
    ) -> Tuple[Thought, "Plan"]:
        prompt_engine = PromptEngine(model)
        system_kwargs = {
            "time": datetime.now().time(),
            "task": task,
            "previous_thought": previous_thought,
            "summary": summary,
            "goals": goals,
            "abilities": abilities,
            "directories": directories,
        }
        system = prompt_engine.load_prompt("new/plan", **system_kwargs)
        LOG.debug("Plan prompt: " + str(system))

        chat_parser = ChatParser(model)
        plan_response = await chat_parser.get_parsed_response(
            system=system,
            containers=[Thought, Plan],
        )
        return plan_response[0], plan_response[1]

    def finished(self):
        """Check if all the goals have been accomplished."""

        for goal in self.goals:
            if not goal.finished():
                return False
        return True

    def get_goal(self):
        for goal in self.goals:
            if not goal.finished():
                return goal
        return None
