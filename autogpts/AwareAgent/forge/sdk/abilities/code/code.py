from forge.helpers.parser import ChatParser
from forge.sdk.abilities.registry import ability
from forge.sdk import ForgeLogger, PromptEngine

from typing import Dict, List


LOG = ForgeLogger(__name__)


@ability(
    name="create_code",
    description="Generates Python code and saves it into a file.",
    parameters=[
        {
            "name": "class_name",
            "description": "A string with the name of the class to create. Empty string if no class should be created.",
            "type": "string",
            "required": True,
        },
        {
            "name": "class_description",
            "description": "The Google-style docstring of the class (if any), including a detailed description, Args, and Returns. Empty string if no class should be created.",
            "type": "string",
            "required": True,
        },
        {
            "name": "functions",
            "description": "A list of dictionaries, where each dictionary contains a function signature as a key and a Google-style docstring as a value.",
            "type": "List[Dict[str, str]]",
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
    class_name: str,
    class_description: str,
    functions: List[Dict[str, str]],
    filename: str,
) -> str:
    """
    Generates a Python class code given the class name, description, functions, and filename. The method
    creates a class definition, writes function signatures with docstrings, and saves the code to the specified file.

    Args:
        class_name (str): The name of the class to create.
        class_description (str): The Google-style docstring of the class, including a detailed description, Args, and Returns.
        functions (List[Dict[str, str]]): A list of dictionaries, where each dictionary contains a function signature as a key and a description as a value.
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

    prompt_engine = PromptEngine(agent.model)
    system_kwargs = {
        "class_name": class_name,
        "class_description": class_description,
        "functions": edited_functions,
    }
    system = prompt_engine.load_prompt("abilities/code", **system_kwargs)

    chat_parser = ChatParser(agent.model)
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
    result = f"Code saved on file {filename}"
    return result
