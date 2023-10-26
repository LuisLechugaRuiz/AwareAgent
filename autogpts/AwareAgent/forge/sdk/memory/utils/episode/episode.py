from datetime import datetime
from pydantic import Field, PrivateAttr
from typing import List, Optional
from forge.helpers.parser.loggable_base_model import LoggableBaseModel
from forge.utils.process_tokens import indent


class Overview(LoggableBaseModel):
    overview: str = Field(
        description="A brief, high-level summary of the episode's content. It provides a quick snapshot of what the episode encompasses without diving into details."
    )


class Episode(LoggableBaseModel):
    _creation_time: str = PrivateAttr(default="")
    _goal: str = PrivateAttr(default="")
    _capability: str = PrivateAttr(default="")
    _ability: str = PrivateAttr(default="")
    _arguments: str = PrivateAttr(default="")
    _observation: str = PrivateAttr(default="")
    _uuid: Optional[str] = PrivateAttr(default=None)
    _child_episodes: List["Episode"] = PrivateAttr(default=[])
    _order: int = PrivateAttr(default=0)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._creation_time = datetime.now().isoformat(timespec="seconds")

    def add_child_episodes(self, episodes: List["Episode"]):
        self._child_episodes.extend(episodes)

    def add_execution(self, ability: str, arguments: str, observation: str):
        self._ability = ability
        self._arguments = arguments
        self._observation = observation

    def get_description(self) -> str:
        """Get a human-readable description of the current goal"""

        return f"Ability: {self._ability}\nArguments: {self._arguments}\nResult: {self._observation}"

    @classmethod
    def get_format(cls, ability, arguments, observation):
        return f"Ability: {ability}\nArguments: {arguments}\nResult: {observation}"

    def link_to_uuid(self, uuid):
        self._uuid = uuid

    def set_order(self, order):
        self._order = order

    def get_uuid(self):
        return self._uuid

    # def get_description(self, include_child_episodes=False):
    #    description = f"- {self._creation_time}: Overview: {self.overview}\nContent: {self.content}"
    #    if self._child_episodes and include_child_episodes:
    #        child_episodes_overview = "\n".join(
    #            [
    #                indent(episode.get_overview().split("\n"))
    #                for episode in self._child_episodes
    #            ]
    #        )
    #        description += f"\nChild Episodes:\n{child_episodes_overview}"
    #    return description

    def to_dict(self):
        return {
            "ability": self._ability,
            "arguments": self._arguments,
            "observation": self._observation,
            "creation_time": self._creation_time,
            "uuid": self._uuid,
            "child_episodes": [episode.to_dict() for episode in self._child_episodes],
            "order": self._order,
        }

    @classmethod
    def from_dict(cls, data):
        episode = Episode()
        episode._creation_time = data.get("creation_time", "")
        episode._ability = data.get("ability", "")
        episode._arguments = data.get("arguments", "")
        episode._observation = data.get("observation", "")
        episode._uuid = data.get("uuid")
        episode._child_episodes = [Episode.from_dict(child_data) for child_data in data.get("child_episodes", [])]
        episode._order = data.get("order", 0)
        episode.link_to_uuid(data["uuid"])
        return episode
