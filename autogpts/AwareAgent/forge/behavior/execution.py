from datetime import datetime
from typing import Any, Dict, Tuple
from pydantic import Field, PrivateAttr

from forge.memory_tmp import Thought
from forge.helpers.parser import ChatParser, LoggableBaseModel, get_json_schema
from forge.sdk import PromptEngine


class Execution(LoggableBaseModel):
    action: str = Field(
        description="The name of the appropriate action that should be used to fulfill the current step. Remember to DON'T REPEAT ACTIONS"
    )
    arguments: Dict[str, Any] = Field(
        description="A dictionary with the action arguments where keys and values are both strings, e.g., {'arg1': 'value1', 'arg2': 'value2'}. You must provide the EXACT arguments (as declared in 'Args' section of each action) with their expected format that the action requires. Failure to do so will prevent the action from executing correctly!"
    )
    _ability: str = PrivateAttr(default=None)

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
        user_kwargs = {
            "schema": get_json_schema([Thought, Execution])
        }
        user = prompt_engine.load_prompt("new/user", **user_kwargs)

        system_kwargs = {
            "time": datetime.now().time(),
            "task": task,
            "goal": goal,
            "previous_thought": previous_thought,
            "ability": ability,
            "summary": summary,
            "directories": directories,
        }
        system = prompt_engine.load_prompt("new/execution", **system_kwargs)

        chat_parser = ChatParser(model)
        action_response = await chat_parser.get_parsed_response(
            system=system,
            user=user,
            containers=[Thought, Execution],
        )
        return action_response[0], action_response[1]

    def get_ability(self) -> str:
        return self._ability

    def set_ability(self, ability: str):
        self._ability = ability
