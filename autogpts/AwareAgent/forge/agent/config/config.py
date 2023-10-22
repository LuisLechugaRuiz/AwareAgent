import json
import os
import shutil
from threading import RLock
from typing import Optional

from forge.sdk.config.config import Config
from forge.sdk.config.storage import get_permanent_storage_path
from forge.agent.config.stage import Stage
from forge.agent.config.status import Status


class AgentConfig(object):
    def __init__(
        self,
        model: str = Config().fast_llm_model,
        stage: Stage = Stage.PLANNING,
        status: Status = Status.WAITING,
    ):
        self.model = model
        self.stage = stage
        self.status = status
        self._stage_lock = RLock()
        self._status_lock = RLock()
        self.save()

    def get_stage(self) -> Stage:
        with self._stage_lock:
            return self.stage

    def set_stage(self, stage: Stage) -> None:
        with self._stage_lock:
            self.stage = stage
            self.save()

    def get_status(self) -> Status:
        with self._status_lock:
            return self.status

    def set_status(self, status: Status) -> None:
        with self._status_lock:
            self.status = status
            self.save()

    @classmethod
    def get_config_file_path(cls) -> str:
        return os.path.join(get_permanent_storage_path(), "config.yaml")

    @classmethod
    def load(cls, folder: str) -> Optional["AgentConfig"]:
        try:
            config_file_path = cls.get_config_file_path(folder)
            with open(config_file_path) as f:
                data = json.load(f)
                return cls.from_json(data)
        except FileNotFoundError:
            return None

    def save(self) -> None:
        config_file = self.get_config_file_path()
        os.makedirs(os.path.dirname(config_file), exist_ok=True)
        with open(config_file, "w") as f:
            f.write(json.dumps(self.to_dict(), indent=2))

    def remove(self) -> bool:
        folder = get_permanent_storage_path()
        if os.path.isdir(folder):
            shutil.rmtree(folder)
            return True
        else:
            return False

    def to_dict(self):
        return {
            "stage": self.stage.name,
            "status": self.status.name,
        }

    @classmethod
    def from_json(cls, data) -> "AgentConfig":
        return AgentConfig(
            stage=Stage[data["stage"]],
            status=Status[data["status"]],
        )
