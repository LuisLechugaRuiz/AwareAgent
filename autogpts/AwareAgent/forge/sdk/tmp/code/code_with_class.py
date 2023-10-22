from forge.helpers.parser import ChatParser
from forge.sdk.abilities.registry import ability
from forge.sdk import PromptEngine
from forge.utils.logger.console_logger import ForgeLogger

from typing import Dict, List
import subprocess

LOG = ForgeLogger(__name__)


@ability(
    name="create_code",
    description="Generates Python code and saves it into a file.",
    parameters=[
        {
            "name": "class_name",
            "description": "A string with the name of the class to create. Empty string if no class should be created.",
            "type": "string",
            "required": False,
        },
        {
            "name": "class_description",
            "description": "The Google-style docstring of the class (if any), including a detailed description, Args, and Returns. Empty string if no class should be created.",
            "type": "string",
            "required": False,
        },
        {
            "name": "functions",
            "description": "A list of dictionaries, where each dictionary contains a function signature as a key and a Google-style docstring as a value.",
            "type": "List[Dict[str, str]]",
            "required": True,
        },
        {
            "name": "main_function_description",
            "description": "A description of the main function and the arguments needed to run the code. Is important you define it explicitely as we will use it to execute the code later.",
            "type": "string",
            "required": True,
        },
        {
            "name": "filename",
            "description": "The name of the file to save the code in.",
            "type": "string",
            "required": True,
        },
    ],
    output_type="str",
)
async def create_code(
    agent,
    task_id: str,
    functions: List[Dict[str, str]],
    main_function_description: str,
    filename: str,
    class_name: str = "",
    class_description: str = "",
) -> str:
    """
    Generates a Python class code given the class name, description, functions, and filename. The method
    creates a class definition, writes function signatures with docstrings, and saves the code to the specified file.

    Args:
        class_name (str): The name of the class to create.
        class_description (str): The Google-style docstring of the class, including a detailed description, Args, and Returns.
        functions (List[Dict[str, str]]): A list of dictionaries, where each dictionary contains a function signature as a key and a description as a value.
        main_function_description (str): A description of the main function and the arguments used to run the code. 
        filename (str): The name of the file to save the code in.

    Returns:
        str: The code of the created class.
    """
    edited_functions = ""
    for function_dict in functions:
        if not isinstance(function_dict, dict):
            return "Error: functions must be a list of dictionaries with the function signature as key and and the description as value."
        for signature, docstring in function_dict.items():
            edited_functions += f"{signature}\n{docstring}\n"

    prompt_engine = PromptEngine(agent.get_model())
    system_kwargs = {
        "class_name": class_name,
        "class_description": class_description,
        "functions": edited_functions,
        "main_function_description": main_function_description,
    }
    system = prompt_engine.load_prompt("abilities/code", **system_kwargs)

    chat_parser = ChatParser(agent.get_model())
    code = await chat_parser.get_response(
        system=system,
        user="Remember to only return the code.",
    )
    LOG.info("Code generated: " + code)
    # Remove triple backticks if they are present at the beginning of the generated code
    if code.startswith("```python"):
        code = code[9:]  # Remove the first 9 characters ("```python\n")
    if code.endswith("```"):
        code = code[:-3]  # Remove the last 3 characters ("```")
    if isinstance(code, str):
        code = code.encode()
    agent.workspace.write(task_id=task_id, path=filename, data=code)
    try:
        await agent.db.create_artifact(
            task_id=task_id,
            file_name=filename,
            relative_path="",
            agent_created=True,
        )
        return f"Successfully saved code on: {filename}"
    except Exception as e:
        return f"Failing to create file: {filename} in the database. Error: {e}"


@ability(
    name="execute_python_file",
    description="Executes a file containing Python code.",
    parameters=[
        {
            "name": "file_name",
            "description": "A string with the name of the file to execute.",
            "type": "string",
            "required": True,
        },
        {
            "name": "args",
            "description": "A list of arguments to pass to the Python script, empty list if no arguments should be passed.",
            "type": "List[str]",
            "required": True,
        },
    ],
    output_type="str",
)
async def execute_python_file(
    agent,
    task_id: str,
    file_name: str,
    args: List[str],
) -> str:
    """Create and execute a Python file in a Docker container and return the STDOUT of the
    executed code. If there is any data that needs to be captured use a print statement

    Args:
        file_name (str): The name of the file to execute
        args (List[str]): A list of arguments to pass to the Python script, empty list if no arguments should be passed.

    Returns:
        str: The STDOUT captured from the code when it ran
    """

    if not str(file_name).endswith(".py"):
        return f"Invalid type of file_name: {file_name}. Only .py files are allowed."
    workspace = agent.workspace.base_path / task_id
    result = subprocess.run(
        ["python", "-B", str(file_name)] + args,
        capture_output=True,
        encoding="utf8",
        cwd=str(workspace),
    )
    if result.returncode == 0:
        return f"Code executed with result: {result.stdout}"
    else:
        return f"Error: {result.stderr}"
