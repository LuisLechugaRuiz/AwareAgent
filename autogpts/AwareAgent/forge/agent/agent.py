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
from forge.sdk.memory.utils.goal import GoalStatus
from forge.sdk.memory.long_term_memory.weaviate import WeaviateMemory
from forge.sdk.behavior import Execution, Plan
from forge.sdk.schema import Status as StepStatus
from forge.utils.logger.file_logger import FileLogger
from forge.utils.directories import list_directories

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
        await self.reset()
        return task

    async def execute_step(self, task_id: str, step_request: StepRequestBody) -> Step:
        task = await self.db.get_task(task_id)
        summary = self.memory.get_episodic_memory()

        step = await sel.run(task, summary)
        # await self.attention_stage(task)  # TODO: Implement me.

        step = await self.execute_stage(task, step_request, summary)
        return step

    async def run(self, task: Task, summary: Optional[str] = None):
        self.config.set_stage(Stage.PLANNING)

        goals = self.memory.get_goals()
        goals_str = None
        if len(goals) > 0:
            goals_str = "\n".join(goal.get_description() for goal in goals)

        # TODO:
        # - Add relevant information from the long term memory.
        thought, plan = await Plan.get_plan(
            task=task.input,
            previous_thought=self.memory.get_thought(),
            abilities=self.abilities.list_abilities_for_prompt(),
            goals=goals_str,
            summary=summary,
            directories=self.get_directories(task_id=task.task_id),
            model=self.config.model,
        )
        self.memory.set_goals(plan.goals)
        self.memory.update_thought(thought)
        
        # Get first goal
        goal = plan.goals[0]
        
        ability = self.verify_ability(goal.ability)
        if ability is None:
            return await self.save_episode(
                task_id, step.step_id, goal.description, goal.ability
            )

        # Execute the ability
        try:
            output = await self.abilities.run_ability(
                task_id, ability, **execution.arguments
            )
        except Exception as e:
            error_output = f"Error executing ability {execution.action} due to: {e}"
            return await self.save_episode(
                task_id,
                step.step_id,
                goal.description,
                execution.action,
                arguments=str(execution.arguments),
                observation=error_output,
            )

        return await self.save_episode(
            task_id,
            step.step_id,
            goal.description,
            execution.action,
            str(execution.arguments),
            str(output),
        )
        
        if self.get_goal() is None:
            self.logger.info("No goals found. Exiting.")
            return False
        return True

    async def execute_stage(
        self, task: Task, step_request: StepRequestBody, summary: str = None
    ):
        self.config.set_stage(Stage.EXECUTION)
        task_id = task.task_id
        step = await self.db.create_step(
            task_id=task_id, input=step_request, is_last=False
        )
        try:
            # TODO: Move to plan. Task will finish with: is_last (flag) or after specific ability (end_task) (back to original idea of ability - end)
            goal = self.get_goal()
            if goal is None:
                self.logger.info("No goals found. Exiting.")
                step.is_last = True
                return step
            if self.memory.max_iterations_reached():
                self.logger.info("Exiting due to too many iterations.")
                step.is_last = True
                return step
            if goal.get_status() == GoalStatus.NOT_STARTED:
                goal.update_status(status=GoalStatus.IN_PROGRESS)
        except Exception as e:
            return await self.save_episode(
                task_id,
                step.step_id,
                goal.description,
                goal.ability,
                observation=f"Wrong goal. Error: {e}",
            )

        ability = self.verify_ability(goal.ability)
        if ability is None:
            return await self.save_episode(
                task_id, step.step_id, goal.description, goal.ability
            )

        # TODO:
        # - Add relevant information from the long term memory.
        thought, execution = await Execution.get_execution(
            task=task.input,
            goal=goal.get_description(),
            previous_thought=self.memory.get_thought().get_description(),
            ability=ability,
            summary=summary,
            directories=self.get_directories(task_id=task_id),
            model=self.config.model,
        )
        self.memory.update_thought(thought)

        new_ability = self.verify_ability(execution.action)
        if new_ability is None:
            return await self.save_episode(
                task_id, step.step_id, goal.description, execution.action
            )

        # Execute the ability
        try:
            output = await self.abilities.run_ability(
                task_id, execution.action, **execution.arguments
            )
        except Exception as e:
            error_output = f"Error executing ability {execution.action} due to: {e}"
            return await self.save_episode(
                task_id,
                step.step_id,
                goal.description,
                execution.action,
                arguments=str(execution.arguments),
                observation=error_output,
            )

        return await self.save_episode(
            task_id,
            step.step_id,
            goal.description,
            execution.action,
            str(execution.arguments),
            str(output),
        )

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
        goal_description: str,
        ability: str,
        arguments: str = "",
        observation: Optional[str] = None,
    ):
        if observation is None:
            observation = f"Ability {ability} not found, verify that it contains only the name of the ability. In case you want to finish the task just set all goal statuses as SUCCEEDED."

        capability = "all_abilities"  # TODO: Divide abilities by classes when the hackathon is over and we want to create general purpose
        await self.memory.add_episode(
            goal=goal_description,
            capability=capability,
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
