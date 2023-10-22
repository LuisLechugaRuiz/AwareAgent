from datetime import datetime
from typing import Any, Dict, Tuple
from pydantic import Field, PrivateAttr

from forge.helpers.parser import ChatParser, LoggableBaseModel
from forge.sdk.memory.utils.thought.thought import Thought
from forge.sdk import PromptEngine
from forge.utils.logger.console_logger import ForgeLogger

LOG = ForgeLogger(__name__)

class Execution(LoggableBaseModel):
    action: str = Field(
        description="The name of the appropriate action that should be used to fulfill the current step. Remember to DON'T REPEAT ACTIONS"
    )
    arguments: Dict[str, Any] = Field(
        description="A dictionary with the action arguments where keys and values are both strings, e.g., {'arg1': 'value1', 'arg2': 'value2'}. You must provide the EXACT arguments (as declared in 'Args' section of each action) with their expected format that the action requires. Failure to do so will prevent the action from executing correctly!"
    )
    _capability: str = PrivateAttr(default="")

    @classmethod
    async def get_execution(
        cls,
        model: str,
        task: str,
        goal: str,
        previous_thought: str,
        ability: str,
        summary: str,
        directories: str,
    ) -> Tuple[Thought, "Execution"]:
        prompt_engine = PromptEngine(model)
        system_kwargs = {
            "date": datetime.now(),
            "task": task,
            "goal": goal,
            "previous_thought": previous_thought,
            "ability": ability,
            "summary": summary,
            "directories": directories,
        }
        system = prompt_engine.load_prompt("new/execution", **system_kwargs)
        # LOG.debug("Execution prompt: " + str(system))

        chat_parser = ChatParser(model)
        action_response = await chat_parser.get_parsed_response(
            system=system,
            containers=[Thought, Execution],
        )
        return action_response[0], action_response[1]

    def get_full_command(self):
        return f"Action: {self.action} with args: {self.arguments}"

    def get_capability(self):
        return self._capability

    def set_capability(self, capability: str):
        self._capability = capability
