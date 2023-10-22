import json
import os
from typing import Dict, List, Optional

# from forge.sdk.agent.communication.events.event import Event -> No events yet!
from forge.sdk.memory.long_term_memory.weaviate import WeaviateMemory
from forge.sdk.memory.utils.goal.goal_memory import GoalMemory
from forge.sdk.memory.utils.goal.goal import Goal
from forge.sdk.memory.utils.episode.episode import Episode
from forge.sdk.memory.utils.episode.episode_manager import EpisodeManager
from forge.sdk.memory.utils.thought.thought import Thought
from forge.sdk.memory.utils.task.task import Task
from forge.sdk.memory.utils.task.task_memory import TaskMemory
from forge.utils.logger.file_logger import FileLogger


class EpisodicMemory:
    def __init__(
        self,
        folder: str,
    ):
        self.file_path = os.path.join(folder, "episodic_memory.json")
        self.folder = folder
        self.logger = FileLogger("episodic_memory")
        loaded = False
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, "r") as f:
                    data = json.load(f)
                    self.load(data=data)
                    loaded = True
                    self.logger.info("Loaded episodic memory")
            except Exception as e:
                self.logger.error(f"Error loading episodic memory: {e}")
        if not loaded:
            self.reset()
        self.save()

    # TODO: add me when using external events (messages, external requests...)
    # def add_event(self, event: Event) -> None:
    #    """Add event to current episode"""
    #
    #    self.events.append(event)
    #
    # def clear_events(self) -> None:
    #    """Clear events"""
    #
    #    self.events = []
    #
    # def get_events(self) -> List[Event]:
    #    """Get events"""
    #
    #    return self.events

    async def add_episode(
        self, goal: str, capability: str, ability: str, arguments: str, observation: str
    ) -> None:
        """Add episode to current episode"""

        await self.episode_manager.create_episodes(
            goal, capability, ability, arguments, observation
        )
        self.save()

    def add_task(self, task: Task):
        """Add a new task"""

        self.task_memory.add_task(task=task)
        self.save()

    def get_thought(self) -> Thought:
        """Get thought"""

        return self.thought

    def get_task(self) -> Optional[Task]:
        """Get task"""

        return self.task_memory.get_task()

    def get_episodes(self) -> List[Episode]:
        """Get previous episodes"""

        return self.episode_manager.get_episodes()

    def get_goals(self) -> List[Goal]:
        """Get goals"""

        return self.goal_memory.get_goals()

    def get_long_term_memory(self) -> WeaviateMemory:

        return self.episode_manager.long_term_memory

    def get_episodic_memory(self) -> Optional[str]:
        """Get episodic memory"""

        episodes = self.episode_manager.get_episodes_str()
        if episodes:
            response = episodes
            summary = self.episode_manager.get_summary_str()
            if summary:
                response += "\nPrevious summary:\n" + summary
            return response
        return None

    def get_similar_episodes(self) -> Dict[str, Episode]:
        """Get most similar episode"""

        return self.similar_episodes

    def get_relevant_information(self) -> str:
        return self.relevant_information

    def max_iterations_reached(self) -> bool:
        """Return True if max iterations reached."""

        result = self.goal_memory.max_iterations_reached()
        # Save as every check increase the number of iterations
        self.save()
        return result

    def set_goals(self, goals: List[Goal]) -> None:
        if len(goals) > 0:
            self.goal_memory.set_goals(goals=goals)
            self.save()

    def update_similar_episodes(self, similar_episodes: Dict[str, Episode]) -> None:
        """Update most similar episode"""

        self.similar_episodes = similar_episodes
        self.save()

    def update_relevant_information(self, relevant_information: str) -> None:
        self.relevant_information = relevant_information
        self.save()

    def update_thought(self, thought: Thought) -> None:
        """Update last thought"""

        self.thought = thought
        self.save()

    def load(self, data):
        """Load the data from a dict"""
        if data["thought"]:
            self.thought = Thought.from_dict(data=data["thought"])
        else:
            self.thought = None
        # self.events = [Event.from_dict(data=event) for event in data["events"]]
        self.episode_manager = EpisodeManager.from_dict(
            data=data["episode_manager"]
        )
        self.goal_memory = GoalMemory.from_dict(
            data=data["goal_memory"]
        )
        self.similar_episodes = {
            str(query): Episode.from_dict(data=episode)
            for query, episode in data["similar_episodes"].items()
        }
        self.task_memory = TaskMemory.from_dict(data["task_memory"])
        self.relevant_information = data["relevant_information"]

    def save(self):
        episodic_memory_dict = self.to_dict()

        with open(self.file_path, "w") as f:
            f.write(json.dumps(episodic_memory_dict, indent=2))

    def to_dict(self):
        return {
            "thought": self.thought.to_dict() if self.thought else None,
            # "events": [event.to_dict() for event in self.events],
            "similar_episodes": {
                query: episode.to_dict() for query, episode in self.similar_episodes.items()
            },
            "goal_memory": self.goal_memory.to_dict(),
            "episode_manager": self.episode_manager.to_dict(),
            "task_memory": self.task_memory.to_dict(),
            "relevant_information": self.relevant_information,
        }

    def set_task_finished(self):
        self.task_memory.set_task_finished()
        self.save()

    async def move_to_long_term_memory(self):
        episodes = self.episode_manager.get_episodes()
        if episodes:
            await self.episode_manager.create_meta_episode(episodes)

    def reset(self):
        self.thought = None
        # self.events: List[Event] = []
        self.goal_memory = GoalMemory()
        self.episode_manager = EpisodeManager()
        self.similar_episodes: Dict[str, Episode] = {}
        self.relevant_information = ""
        # TODO: We can do a priority queue but we need extra logic to evalute the priority, priority should be part of Task - Task is managed by Notion or UI.
        self.task_memory = TaskMemory()
        self.save()
        self.logger.info("Starting new memory")
