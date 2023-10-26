from forge.sdk import (
    Agent,
    AgentDB,
    Step,
    StepRequestBody,
    Task,
    TaskRequestBody,
    Workspace,
)
from forge.agent.config import AgentConfig, Stage, Status
from forge.sdk.config.storage import get_permanent_storage_path
from forge.sdk.memory.short_term_memory.episodic_memory import EpisodicMemory
from forge.sdk.memory.long_term_memory.weaviate import WeaviateMemory
from forge.sdk.behavior import Plan
from forge.sdk.schema import Status as StepStatus
from forge.utils.logger.file_logger import FileLogger
from forge.utils.directories import list_directories

from datetime import datetime
from typing import Optional
import os


class ForgeAgent(Agent):
    def __init__(self, database: AgentDB, workspace: Workspace):
        """
        The database is used to store tasks, steps and artifact metadata. The workspace is used to
        store artifacts. The workspace is a directory on the file system.

        Feel free to create subclasses of the database and workspace to implement your own storage
        """
        super().__init__(database, workspace)
        self.config = AgentConfig("gpt-4")  # TODO: Get from cfg.
        self.logger = FileLogger("main")
        self.memory = EpisodicMemory(
            folder=get_permanent_storage_path()
        )
        self.task_start_time = datetime.now()

    async def create_task(self, task_request: TaskRequestBody) -> Task:
        """
        The agent protocol, which is the core of the Forge, works by creating a task and then
        executing steps for that task. This method is called when the agent is asked to create
        a task.

        We are hooking into function to add a custom log message. Though you can do anything you
        want here.
        """
        task = await super().create_task(task_request)
        self.config.set_status(Status.ACTIVE)
        self.logger.info(
            f"📦 Task created: {task.task_id} input: {task.input[:40]}{'...' if len(task.input) > 40 else ''}"
        )
        self.task_start_time = datetime.now()
        await self.reset()
        return task

    async def execute_step(self, task_id: str, step_request: StepRequestBody) -> Step:
        task = await self.db.get_task(task_id)
        summary = self.memory.get_episodic_memory()
        step = await self.run(task, step_request, summary)
        return step

    async def run(self, task: Task, step_request: StepRequestBody, summary: Optional[str] = None):
        self.config.set_stage(Stage.PLANNING)
        task_id = task.task_id
        step = await self.db.create_step(
            task_id=task_id, input=step_request, is_last=False
        )
        step_id = step.step_id
        # Verify if max iterations
        if self.memory.max_iterations_reached():
            self.logger.info("Exiting due to too many iterations.")
            step.is_last = True
            return step

        # TODO:
        # - Add relevant information from the long term memory.
        execution = await Plan.get_plan(
            task=task.input,
            previous_thought=self.memory.get_thought(),
            abilities=self.abilities.list_abilities_for_prompt(),
            summary=summary,
            directories=self.get_directories(task_id=task.task_id),
            model=self.config.model,
        )
        self.memory.update_thought(execution.reasoning)

        # Verify that ability is valid
        ability = self.verify_ability(execution.ability)
        if ability is None:
            return await self.save_episode(
                task_id, step_id, execution.ability
            )

        # Execute the ability
        try:
            output = await self.abilities.run_ability(
                task_id, execution.ability, **execution.arguments
            )
        except Exception as e:
            error_output = f"Error executing ability {execution.ability} due to: {e}"
            return await self.save_episode(
                task_id,
                step_id,
                execution.ability,
                arguments=str(execution.arguments),
                observation=error_output,
            )
        step = await self.save_episode(
            task_id,
            step_id,
            execution.ability,
            str(execution.arguments),
            str(output),
        )
        if execution.ability == "finish":
            self.logger.info("Finishing finish ability")
            step.is_last = True
        if execution.is_last:
            self.logger.info("Finishing due to last execution.")
            step.is_last = True
        if step.is_last:
            task_time_sec = (datetime.now() - self.task_start_time).total_seconds()
            self.logger.info(f"Task took {task_time_sec} seconds to complete.")
        return step

    def get_short_term_memory(self) -> EpisodicMemory:
        return self.memory

    def get_long_term_memory(self) -> WeaviateMemory:
        return self.memory.get_long_term_memory()

    def get_goal(self):
        for goal in self.memory.get_goals():
            try:
                if not goal.finished():
                    return goal
            except Exception as e:
                goal.description += f"- Wrong goal status: {e}. Remember it should be one of the following: NOT_STARTED, IN_PROGRESS, SUCCEEDED, FAILED"
                return goal
        return None

    def get_directories(self, task_id: str):
        path = (self.workspace.base_path / task_id).resolve()
        if not os.path.exists(path):
            os.makedirs(path)
        return list_directories(path)

    def verify_ability(self, ability: str):
        try:
            ability = self.abilities.abilities[ability]
            return ability
        except Exception:
            return None

    async def reset(self):
        await self.memory.move_to_long_term_memory()
        self.memory.reset()

    async def save_episode(
        self,
        task_id: str,
        step_id: str,
        ability: str,
        arguments: str = "",
        observation: Optional[str] = None,
    ):
        if observation is None:
            observation = f"Ability {ability} not found, verify that it contains only the name of the ability. In case you want to finish the task just set all goal statuses as SUCCEEDED."

        await self.memory.add_episode(
            ability=ability,
            arguments=arguments,
            observation=observation,
        )

        step = await self.db.update_step(
            task_id,
            step_id,
            status=StepStatus.completed.value,
            output=observation,
        )
        return step

    def get_model(self):
        return self.config.model
