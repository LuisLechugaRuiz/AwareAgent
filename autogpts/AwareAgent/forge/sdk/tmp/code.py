from typing import List, Tuple, Type
from pydantic import Field
import subprocess
import re

from forge.helpers.parser import ChatParser, LoggableBaseModel
from forge.sdk.abilities.registry import ability
from forge.sdk import PromptEngine
from forge.utils.logger.file_logger import FileLogger


class Debug(LoggableBaseModel):
    reasoning: str = Field(
        description="The logical explanation or analysis that identifies the problem.",
    )
    error_in_code: bool = Field(
        description="A flag indicating if the error is in the code. True represents an error in the code, while False indicates an error in the test.",
    )
    improvement: str = Field(
        description="Detailed description of the suggested fix or enhancement to resolve the error.",
    )


@ability(
    name="create_and_validate_code",
    description="Generates Python code, saves it into a file, and validates its correctness through robust internal tests. When calling this function send always the full task, don't try to manage it yourself.",
    parameters=[
        {
            "name": "filename",
            "description": "The name of the file to save the code in.",
            "type": "string",
            "required": True,
        },
        {
            "name": "task",
            "description": "A complete transcription of the assignment, meticulously outlining all requirements, functions, and expected outcomes as stated, without omission. This serves as the authoritative directive for code development, demanding strict adherence to the task's full scope and minute details.",
            "type": "string",
            "required": True,
        },
        {
            "name": "functions",
            "description": "A list with the function signature. Explain them on the task parameter.",
            "type": "List[str]",
            "required": True,
        },
        {
            "name": "validation",
            "description": "A detailed explanation of the specific requirements and benchmarks that the code must successfully meet to be deemed correct. Add all the information needed as this is used to generate the tests.",
            "type": "string",
            "required": True,
        },
    ],
    output_type="str",
)
async def create_and_validate_code(
    agent,
    task_id: str,
    filename: str,
    task: str,
    functions: str,
    validation: str,
) -> str:
    # TODO:
    # 1. Class addition. Not only functions, but also classes.
    # 2. Manager to evaluate description and translate it into functions | Validation into tests.
    # 3. Improve code should focus on the errors, we should find a way to send only partial code..
    logger = FileLogger("code")
    model = agent.get_model()

    # 1. Create code
    code = await get_code(
        model=model,
        task=task,
        functions=functions,
        validation=validation,
    )
    code = clean_code(code)
    agent.workspace.write(task_id=task_id, path=filename, data=code.encode("utf-8"))
    logger.info(f"Code: {code}")
    # 2. Create tests
    tests = await get_tests(
        model=model, code=code, validation=validation, filename=filename
    )
    tests = clean_code(tests)
    logger.info(f"Tests: {tests}")
    filename_test = filename.replace(".py", "_test.py")
    agent.workspace.write(
        task_id=task_id, path=filename_test, data=tests.encode("utf-8")
    )
    retries = 5
    while retries > 0:
        # 3. Execute tests
        result, success = execute_python_code(agent, task_id, filename_test, args=[])
        logger.info(f"Result: {result}")
        if success:
            await agent.db.create_artifact(
                task_id=task_id,
                file_name=filename,
                relative_path="",
                agent_created=True,
            )
            logger.info("The code passed the tests.")
            return f"Code saved successfully at: {filename}. Tests passed successfully. Please do not execute it, as it has already been verified. Set all goals in task to SUCCEEDED!"
        # 4. Debug the error and determine a fix
        logger.info("The code didn't pass the tests. debugging...")
        debug_response = await debug_code(
            model=model, description=task, code=code, tests=tests, result=result
        )
        logger.info("Trying to improve the code...")

        # 5. Fix code or test
        response = await improve_code(
            model=model,
            description=task,
            code=code,
            tests=tests,
            result=result,
            improvement=debug_response.improvement,
            error_in_code=debug_response.error_in_code,
        )
        code = clean_code(response)
        agent.workspace.write(task_id=task_id, path=filename, data=code.encode("utf-8"))
        logger.info(f"Improved code: {code}")
        retries -= 1
    await agent.db.create_artifact(
        task_id=task_id,
        file_name=filename,
        relative_path="",
        agent_created=True,
    )
    return f"Code saved successfully at: {filename}. Tests failed. Some tests might be broken, review the code manually and set as SUCCEEDED if it is correct."


def preproccess(functions):
    edited_functions = ""
    for function_dict in functions:
        if not isinstance(function_dict, dict):
            return "Error: functions must be a list of dictionaries with the function signature as key and and the description as value."
        for signature, docstring in function_dict.items():
            edited_functions += f"{signature}\n{docstring}\n"
    return edited_functions


def clean_code(text):
    # Define the pattern to match
    pattern = r"```python(.*?)```"

    # Search for the pattern in the text
    match = re.search(
        pattern, text, re.DOTALL
    )  # re.DOTALL allows the dot (.) to match newlines as well

    if match:
        # If a match is found, extract it. The actual code is in the first group (group 1)
        # Also, strip any leading/trailing whitespace, including the newline that follows ```python
        code = match.group(1).strip()
    else:
        code = text
    return code


# This assumes that description + code + tests is not too long.. might need reconsideration.
async def improve_code(
    model: str,
    description: str,
    code: str,
    tests: str,
    result: str,
    improvement: str,
    error_in_code: bool = True,
):
    prompt_engine = PromptEngine(model)
    system_kwargs = {
        "description": description,
        "code": code,
        "tests": tests,
        "result": result,
        "improvement": improvement,
        "error_in_code": error_in_code,
    }
    system = prompt_engine.load_prompt("abilities/code/improve_code", **system_kwargs)
    return await get_response(model=model, system=system)


async def debug_code(
    model: str,
    description: str,
    code: str,
    tests: str,
    result: str,
):
    prompt_engine = PromptEngine(model)
    system_kwargs = {
        "description": description,
        "code": code,
        "tests": tests,
        "result": result,
    }
    system = prompt_engine.load_prompt("abilities/code/debug", **system_kwargs)
    return await get_complex_response(model=model, system=system, containers=[Debug])


async def get_tests(model: str, code: str, validation: str, filename: str) -> str:
    prompt_engine = PromptEngine(model)
    system_kwargs = {
        "code": code,
        "validation": validation,
        "filename": filename,
    }
    system = prompt_engine.load_prompt("abilities/code/test", **system_kwargs)
    return await get_response(model=model, system=system)


async def get_code(model: str, task: str, functions: str, validation: str) -> str:
    prompt_engine = PromptEngine(model)
    system_kwargs = {
        "task": task,
        "functions": functions,
        "validation": validation,
    }
    system = prompt_engine.load_prompt("abilities/code/code", **system_kwargs)
    # TODO: Check if we need to use containers or just ask for raw code.
    return await get_response(model=model, system=system)


async def get_response(model: str, system: str) -> str:
    chat_parser = ChatParser(model)
    return await chat_parser.get_response(
        system=system,
        user="Remember to only return the info requested. Don't add any extra information.",
    )


async def get_complex_response(
    model: str,
    system: str,
    containers: List[Type[LoggableBaseModel]] = [],
    index: int = 0,
) -> Type[LoggableBaseModel]:
    chat_parser = ChatParser(model)
    response = await chat_parser.get_parsed_response(
        system=system,
        containers=containers,
    )
    return response[index]


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
    result, success = execute_python_code(agent, task_id, file_name, args)
    return result


def execute_python_code(
    agent,
    task_id: str,
    file_name: str,
    args: List[str],
) -> Tuple[str, bool]:
    # TODO: GET CURRENT ARTIFACTS. VERIFY AT END AND ADD NEW ARTIFACTS (IN CASE CODE GENERATED ANY!)

    if not str(file_name).endswith(".py"):
        return (
            f"Invalid type of file_name: {file_name}. Only .py files are allowed.",
            False,
        )
    workspace = agent.workspace.base_path / task_id
    # TODO: Move to docker container
    result = subprocess.run(
        ["python", "-B", str(file_name)] + args,
        capture_output=True,
        encoding="utf8",
        cwd=str(workspace),
    )
    if result.returncode == 0:
        return f"Code executed with result: {result.stdout}", True
    else:
        return f"Error: {result.stderr}", False
