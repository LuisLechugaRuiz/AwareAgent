from enum import Enum


class TaskStatus(Enum):
    NOT_STARTED = "Not Started"
    IN_PROGRESS = "In Progress"
    DONE = "Done"

    @classmethod
    def from_string(cls, string_value: str):
        try:
            return cls(string_value)
        except ValueError:
            raise ValueError(f"Invalid string value: {string_value}")


def main():
    print(TaskStatus.from_string("Not started").name)


if __name__ == "__main__":
    main()
