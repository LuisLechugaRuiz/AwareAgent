import os
import toml
from pathlib import Path


# TODO: Adapt this function to get from parent now that we are updating docker to dr.
def get_permanent_storage_path() -> Path:
    # Find pyproject.toml
    current_path = Path(__file__).resolve()
    while current_path != Path("/") and not (current_path / "pyproject.toml").is_file():
        current_path = current_path.parent

    if not (current_path / "pyproject.toml").is_file():
        raise FileNotFoundError("Cannot find pyproject.toml.")

    # Read pyproject.toml
    pyproject_data = toml.load(current_path / "pyproject.toml")

    # Check if the project is a Poetry project
    if "tool" not in pyproject_data or "poetry" not in pyproject_data["tool"]:
        raise ValueError("This is not a Poetry project.")

    # Get the project root (absolute path)
    project_root = current_path

    # Construct the path to the permanent_storage folder (absolute path)
    permanent_storage_path = project_root / "permanent_storage"

    # Check if the permanent_storage folder exists, create it if it doesn't
    permanent_storage_path.mkdir(parents=True, exist_ok=True)
    return permanent_storage_path
