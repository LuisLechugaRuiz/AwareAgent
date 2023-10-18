from pydantic import BaseModel


class LoggableBaseModel(BaseModel):
    def __str__(self):
        attrs = vars(self)
        return f"---{self.__class__.__name__}---" + "\n".join(
            [f"\n{k}:\n{v}" for k, v in attrs.items() if k != "__dict__"]
        )
