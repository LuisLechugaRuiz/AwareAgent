from typing import List, Tuple, Type
from pydantic import Field
import subprocess
import re

from forge.helpers.parser import ChatParser, LoggableBaseModel
from forge.sdk.abilities.registry import ability
from forge.sdk.config.config import Config
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
    name="modify_code",
    description=(
        "Refines existing code by replacing a specified segment with improved logic, ensuring precision enhancements while maintaining the overall code structure."
    ),
    parameters=[
        {
            "name": "filename",
            "description": "The file where the code segment resides, ensuring proper location of the modification.",
            "type": "string",
            "required": True,
        },
        {
            "name": "description",
            "description": "Guides the modification process, clarifying the desired functionality improvement.",
            "type": "string",
            "required": True,
        },
        {
            "name": "new_code",
            "description": "The enhanced code snippet intended to replace the existing segment.",
            "type": "string",
            "required": True,
        },
        {
            "name": "pattern",
            "description": "The regular expression pattern that matches the code segment to be replaced. This pattern must be accurate and specific to prevent unintended modifications.",
            "type": "string",
            "required": True,
        },
        {
            "name": "validation",
            "description": "Criteria confirming the successful integration and functionality of the new code snippet.",
            "type": "string",
            "required": True,
        },
    ],
    output_type="str",
)
async def modify_code(
    agent,
    task_id: str,
    filename: str,
    description: str,
    new_code: str,
    pattern: str,
    validation: str,
) -> str:
    logger = FileLogger("code")

    # Reading the existing code
    try:
        logger.info(f"Loading code from {filename}")
        existing_code_bytes = await agent.workspace.read(task_id=task_id, path=filename)
        existing_code = existing_code_bytes.decode("utf-8")
    except Exception as e:
        return f"Error reading the code file: {str(e)}"

    # Attempting the regex replacement
    try:
        updated_code = re.sub(pattern, new_code, existing_code, flags=re.DOTALL)
    except re.error as e:
        return f"Regex error: {str(e)}"

    # Check if any modification was made
    if existing_code == updated_code:
        return "No match found or replacement failed. Check the accuracy of the regex pattern."

    # Writing the new code back to the workspace
    try:
        write_to_file(
            agent=agent,
            task_id=task_id,
            filename=filename,
            code=updated_code.encode("utf-8"),
            overwrite=True,
        )
        logger.info(f"Code updated in {filename}")
    except Exception as e:
        return f"Error writing the updated code back to the file: {str(e)}"

    # Delegate to the testing and improvement function
    return await test_and_improve_code(
        agent, task_id, filename, description, updated_code, validation
    )


@ability(
    name="create_code",
    description=(
        "Entrusted with a code snippet, this ability takes an active role in its refinement. It starts with the code provided, then probes it through a series of evaluations, enhancements, and rigorous testing, ensuring the code not only works but is efficient and meets established standards. This is not about writing new code from scratch; instead, it's about taking something existing and making it better, ensuring it's worthy of inclusion in the system's repository."
    ),
    parameters=[
        {
            "name": "filename",
            "description": "The designated file name where the enhanced code will reside, facilitating organization and retrieval.",
            "type": "string",
            "required": True,
        },
        {
            "name": "description",
            "description": "A narrative that outlines the code's purpose and expected functionality, serving as the guidepost for its refinement journey.",
            "type": "string",
            "required": True,
        },
        {
            "name": "code",
            "description": "The raw material of the process: the initial code snippet that this ability will scrutinize and enhance.",
            "type": "string",
            "required": True,
        },
        {
            "name": "validation",
            "description": "The benchmarks and standards that the polished code must satisfy, defining the quality and functionality bar it needs to meet.",
            "type": "string",
            "required": True,
        },
        {
            "name": "overwrite",
            "description": "Determines whether the content within the existing file will be replaced or if the new, refined code will be appended to it. By default, it overwrites to ensure the most updated version is what's stored.",
            "type": "bool",
            "required": False,
            "default": True,
        },
    ],
    output_type="str",
)
async def create_code(
    agent,
    task_id: str,
    filename: str,
    description: str,
    code: str,
    validation: str,
    overwrite: bool = True,
) -> str:
    logger = FileLogger("code")

    # Save initial code
    code = clean_code(code)
    logger.info(f"Code: {code}")
    write_to_file(agent, task_id, filename, code, overwrite)

    return await test_and_improve_code(
        agent, task_id, filename, description, code, validation, overwrite
    )


async def test_and_improve_code(
    agent,
    task_id: str,
    filename: str,
    description: str,
    code: str,
    validation: str,
    overwrite: bool = True,
) -> str:
    logger = FileLogger("code")
    model = agent.get_model()

    # 1. Create tests
    tests = await get_tests(
        model=model, code=code, validation=validation, filename=filename
    )
    tests = clean_code(tests)
    logger.info(f"Tests: {tests}")
    filename_test = filename.replace(".py", "_test.py")
    write_to_file(agent, task_id, filename_test, tests, overwrite)

    retries = Config().max_improvement_retries
    code_improvements = ""

    success = False
    final_result = None

    while retries > 0 and not success:
        # 2. Execute tests
        result, success = execute_python_code(agent, task_id, filename_test, args=[])
        logger.info(f"Result: {result}")
        if success:
            final_result = result
        else:
            # 3. Debug the error and determine a fix
            logger.info("The code didn't pass the tests. debugging...")
            debug_response = await debug_code(
                model=model,
                description=description,
                code=code,
                tests=tests,
                result=result,
            )
            error_in_code = debug_response.error_in_code
            improvement = debug_response.improvement

            # 4. Fix code or test
            if error_in_code:
                logger.info("Trying to improve the code...")
            else:
                logger.info("Trying to improve the tests...")
            response = await improve_code(
                model=model,
                description=description,
                code=code,
                tests=tests,
                result=result,
                improvement=improvement,
                error_in_code=error_in_code,
            )
            improved_code = clean_code(response)
            if error_in_code:
                code = improved_code
                code_improvements += f"-Improvement: {improvement}\n"
                write_to_file(
                    agent=agent,
                    task_id=task_id,
                    filename=filename,
                    code=code,
                    overwrite=overwrite,
                )
                logger.info(f"Improved code: {code}")
            else:
                tests = improved_code
                write_to_file(
                    agent=agent,
                    task_id=task_id,
                    filename=filename_test,
                    code=tests,
                    overwrite=overwrite,
                )
                logger.info(f"Improved tests: {tests}")
            retries -= 1

    logger.info(f"Final code: {code}")
    # Upload artifact
    await agent.db.create_artifact(
        task_id=task_id,
        file_name=filename,
        relative_path="",
        agent_created=True,
    )

    if not success:
        # Execute the tests one last time to get the final result.
        final_result, success = execute_python_code(
            agent, task_id, filename_test, args=[]
        )
        logger.info(f"Final execution result: {final_result}")

    # Prepare the base response.
    if success:
        response_message = f"Tests passed successfully!! Code:\n{code}\nThe code has been saved at: {filename}."
    else:
        response_message = f"Tests failed... Code:\n{code}\nThe code has been saved at: {filename}. Last error:\n{final_result}"

    # Append information about the improvements, if any.
    if code_improvements:
        response_message += f"\nCode improvements:\n{code_improvements}"

    return response_message


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


def write_to_file(agent, task_id, filename, code, overwrite=True):
    if overwrite:
        agent.workspace.write(task_id=task_id, path=filename, data=code.encode("utf-8"))
    else:
        file_path = agent.workspace._resolve_path(task_id=task_id, path=filename)
        with open(file_path, "ab") as f:
            f.write(code.encode("utf-8"))


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
