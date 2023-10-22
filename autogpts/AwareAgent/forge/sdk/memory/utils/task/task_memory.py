from typing import List, Optional
import threading

from forge.sdk.memory.utils.task.task import Task


class TaskMemory:
    def __init__(self):
        self.tasks: List[Task] = []
        self.lock = threading.Lock()

    def add_task(self, task: Task) -> None:
        """Add task"""

        with self.lock:
            self.tasks.append(task)

    def get_task(self) -> Optional[Task]:
        """Get task or None"""

        with self.lock:
            if len(self.tasks) > 0:
                return self.tasks[0]
            return None

    def clear(self) -> None:
        """Clear current step"""

        with self.lock:
            self.tasks.clear()

    def to_dict(self):
        return {
            "tasks": [task.to_dict() for task in self.tasks],
        }

    def set_task_finished(self):
        """Set current task as finished"""

        with self.lock:
            if len(self.tasks) > 0:
                self.tasks.pop(0)

    @classmethod
    def from_dict(cls, data):
        task_memory = cls()
        for task in data["tasks"]:
            task_memory.add_task(Task.from_dict(data=task))
        return task_memory
