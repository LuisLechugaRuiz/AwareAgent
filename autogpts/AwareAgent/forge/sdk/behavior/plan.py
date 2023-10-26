from datetime import datetime
from pydantic import Field

from forge.sdk.behavior.execution import Execution
from forge.sdk.memory.utils.goal.goal import Goal
from forge.helpers.parser import ChatParser, LoggableBaseModel
from forge.sdk import PromptEngine
from forge.utils.logger.console_logger import ForgeLogger

LOG = ForgeLogger(__name__)


class Plan(LoggableBaseModel):
    goals: list[Goal] = Field(
        description="The updated list of goals that you should work on, consider previous goals and update them based on new information"
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
    ) -> Execution:
        prompt_engine = PromptEngine(model)
        system_kwargs = {
            "date": datetime.now(),
            "task": task,
            "previous_thought": previous_thought,
            "summary": summary,
            "abilities": abilities,
            "directories": directories,
        }
        system = prompt_engine.load_prompt("new/plan", **system_kwargs)
        LOG.debug("Plan prompt: " + str(system))

        chat_parser = ChatParser(model)
        plan_response = await chat_parser.get_parsed_response(
            system=system,
            containers=[Execution],
        )
        return plan_response[0]

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
