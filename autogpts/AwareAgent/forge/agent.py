from forge.sdk import (
    Agent,
    AgentDB,
    ForgeLogger,
    Step,
    StepRequestBody,
    Task,
    TaskRequestBody,
    Workspace,
)
from forge.sdk.db import StepModel

# Additions
from forge.behavior import Execution, Plan
from forge.helpers.stages import Stage
from forge.memory_tmp.goal import Goal, GoalStatus
from forge.sdk.schema import Status as StepStatus
from forge.utils.directories import list_directories

from typing import List
import os

LOG = ForgeLogger(__name__)


class ForgeAgent(Agent):
    def __init__(self, database: AgentDB, workspace: Workspace):
        """
        The database is used to store tasks, steps and artifact metadata. The workspace is used to
        store artifacts. The workspace is a directory on the file system.

        Feel free to create subclasses of the database and workspace to implement your own storage
        """
        super().__init__(database, workspace)
        self.stage = Stage.PLANNING
        self.model = "gpt-4"  # TODO: Move to config
        self.last_thought = ""  # TODO: Move to memory
        self.goals: List["Goal"] = []  # TODO: Move to memory
        self.iterations = 0  # TODO: REMOVE - JUST A SAFE GUARD TO AVOID LOOPING

    async def create_task(self, task_request: TaskRequestBody) -> Task:
        """
        The agent protocol, which is the core of the Forge, works by creating a task and then
        executing steps for that task. This method is called when the agent is asked to create
        a task.

        We are hooking into function to add a custom log message. Though you can do anything you
        want here.
        """
        task = await super().create_task(task_request)
        LOG.info(
            f"📦 Task created: {task.task_id} input: {task.input[:40]}{'...' if len(task.input) > 40 else ''}"
        )
        self.reset()
        return task

    async def execute_step(self, task_id: str, step_request: StepRequestBody) -> Step:
        task = await self.db.get_task(task_id)
        steps, _ = await self.db.list_steps(task.task_id)
        summary = None
        if len(steps) > 0:
            summary = self.get_summary(steps)

        await self.plan_stage(task, steps, summary)
        goal = self.get_goal()
        if goal is None:
            LOG.info("No goals found. Exiting.")
            steps[-1].is_last = True
            return steps[-1]
        if self.iterations > 5:
            LOG.info("Exiting due to too many iterations.")
            steps[-1].is_last = True
            return steps[-1]

        # await self.attention_stage(task)  # TODO: Implement me.

        step = await self.execute_stage(task, goal, step_request, summary)
        self.iterations += 1
        return step

    async def plan_stage(self, task, steps, summary=None):
        # STOP!
        LOG.debug("Steps: " + str(steps))
        goals_str = None
        if len(self.goals) > 0:
            goals_str = "\n".join(goal.get_description() for goal in self.goals)

        # TODO:
        # - Add relevant information from the long term memory.
        thought, plan = await Plan.get_plan(
            task=task.input,
            previous_thought=self.last_thought,
            abilities=self.abilities.list_abilities_for_prompt(),
            goals=goals_str,
            summary=summary,
            directories=self.get_directories(task_id=task.task_id),
            model=self.model,
        )
        self.goals = plan.goals
        self.last_thought = thought
        # LOG.debug("Plan: " + str(plan))
        # self.stage = Stage.EXECUTION  # TODO: Move to attention.
        if self.get_goal() is None:
            LOG.info("No goals found. Exiting.")
            return False
        return True

    async def execute_stage(self, task: Task, goal: Goal, step_request: StepRequestBody, summary: str = None):
        task_id = task.task_id

        if goal.get_status() == GoalStatus.NOT_STARTED:
            goal.update_status(status=GoalStatus.IN_PROGRESS)

        step = await self.db.create_step(
            task_id=task_id, input=step_request, is_last=False
        )
        try:
            ability = self.abilities.abilities[goal.ability]
        except Exception as e:
            additional_input = {
                "goal": goal.description,
                "ability": goal.ability,
                "arguments": "",
            }
            step = await self.db.update_step(
                task_id,
                step.step_id,
                status=StepStatus.completed.value,
                additional_input=additional_input,
                output=f"Ability {goal.ability} not found, verify that it contains only the name of the ability. Error: {e}",
            )
            return step

        # TODO:
        # - Add relevant information from the long term memory.
        thought, execution = await Execution.get_execution(
            task=task.input,  # TODO: Do we need to convert it?
            goal=goal,
            previous_thought=self.last_thought,
            ability=ability,
            summary=summary,
            directories=self.get_directories(task_id=task_id),
            model=self.model,
        )
        self.last_thought = thought

        # Create a new step in the database
        output = await self.abilities.run_ability(
            task_id, ability.name, **execution.arguments
        )
        LOG.debug("Output: " + str(output))
        additional_input = {
            "goal": goal.description,
            "ability": ability.name,
            "arguments": execution.arguments,
        }
        # TODO: Update step with additional output -> CREATE A CLASS ADDITIONAL OUTPUT WITH OVERVIEW + SUMMARY
        step = await self.db.update_step(
            task_id,
            step.step_id,
            status=StepStatus.completed.value,
            additional_input=additional_input,
            output=output,
        )
        return step

    def get_summary(self, steps: List[StepModel], show_uuid: bool = True):
        summary = ""
        for step in steps[::-1]:
            goal = step.additional_input["goal"]
            ability = step.additional_input["ability"]
            arguments = step.additional_input["arguments"]
            current_step_summary = f"Goal: {goal}\nAbility: {ability}\nArguments: {arguments}\nResult: {step.output}\n\n"
            summary += current_step_summary

        return summary

    def get_goal(self):
        for goal in self.goals:
            if not goal.finished():
                return goal
        return None

    # TODO: Verify that tasks are not connected!
    def reset(self):
        self.last_thought = ""
        self.goals = []
        self.iterations = 0

    def get_directories(self, task_id: str):
        path = (self.workspace.base_path / task_id).resolve()
        if not os.path.exists(path):
            os.makedirs(path)
        return list_directories(path)
