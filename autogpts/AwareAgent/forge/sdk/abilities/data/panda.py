import csv
import pandas as pd
from pandasai import Agent as PandasAgent
from pandasai.smart_dataframe import SmartDataframe
from pandasai.llm.openai import OpenAI
from pydantic import Field
from typing import Tuple, List
from io import StringIO

from forge.sdk.abilities.registry import ability
from forge.sdk import PromptEngine
from forge.utils.logger.file_logger import FileLogger
from forge.helpers.parser import ChatParser, LoggableBaseModel


class Answer(LoggableBaseModel):
    reasoning: str = Field(
        description="Detailed rationale outlining the analytical process used to determine the appropriateness of the answer in relation to the question posed."
    )
    fix: str = Field(
        description="Specify the error and suggest a fix to address the issue identified in the reasoning. Empty if the answer is already valid."
    )
    validated: bool = Field(
        description="Confirmation status as to whether the provided answer accurately and sufficiently addresses the presented question."
    )


class ShouldShowFullDfs(LoggableBaseModel):
    reasoning: str = Field(
        description="Explanation of the analysis leading to the decision of whether to show full dfs or not."
    )
    should_show_full_dfs: bool = Field(description="Flag to set if full dfs should be shown.")


@ability(
    name="process_csv_and_save",
    description="Processes CSVs using natural language queries and saves the output.",
    parameters=[
        {
            "name": "input_filenames",
            "description": "The names of the CSV files containing the structured data to analyze.",
            "type": "List[string]",
            "required": True,
        },
        {
            "name": "output_filename",
            "description": "The name of the CSV file to save the result.",
            "type": "string",
            "required": True,
        },
        {
            "name": "request",
            "description": "The request that will be used to process the data and save the result.",
            "type": "string",
            "required": True,
        },
    ],
    output_type="str",
)
async def process_csv_and_save(
    agent, task_id: str, input_filenames: List[str], output_filename: str, request: str
) -> str:
    response = await pandasai_chatbot(
        agent, task_id, input_filenames, request
    )
    # Check if the response is a DataFrame before saving
    if isinstance(response, SmartDataframe):
        # Save DataFrame to CSV
        output_filepath = agent.workspace.base_path / task_id / output_filename
        dataframe = response.original_import()
        dataframe.to_csv(output_filepath, index=False)
    else:
        # Save response to file
        response = str(response).encode("utf-8")
        agent.workspace.write(task_id=task_id, path=output_filename, data=response)
    await agent.db.create_artifact(
        task_id=task_id,
        file_name=output_filename,
        relative_path="",
        agent_created=True,
    )
    return f"Data saved successfully at: {output_filename}"


@ability(
    name="get_insights_from_csv",
    description="Derives insights from CSVs using natural language.",
    parameters=[
        {
            "name": "filenames",
            "description": "The names of the CSV files containing the structured data to analyze.",
            "type": "List[string]",
            "required": True,
        },
        {
            "name": "question",
            "description": "A question referring to the data's content, which the system will interpret and respond to with the appropriate data insights.",
            "type": "string",
            "required": True,
        },
    ],
    output_type="str",
)
async def get_insights_from_csv(
    agent, task_id: str, filenames: List[str], question: str
) -> str:
    response_str = await pandasai_chatbot(agent, task_id, filenames, question)
    return response_str


async def pandasai_chatbot(
    agent,
    task_id: str,
    filepaths: List[str],
    question: str,
) -> Tuple[str, bool]:
    logger = FileLogger("pandasai_chatbot")
    dfs = []
    full_processed_data = ""
    for file in filepaths:
        model = agent.get_model()
        raw_data = agent.workspace.read(task_id=task_id, path=file).decode("utf-8")
        try:
            processed_data = preprocess_csv(raw_data)
        except ValueError as e:
            logger.error(f"Error: {e}")
            processed_data = raw_data
        full_processed_data += processed_data
        csv_buffer = StringIO(processed_data)
        dfs.append(pd.read_csv(csv_buffer))

    show_full_dfs = await preprocess_request(model, question, full_processed_data)
    logger.info(f"show_full_dfs: {show_full_dfs}")

    llm = OpenAI()
    llm.model = model
    panda_agent = PandasAgent(
        dfs,
        config={"llm": llm, "verbose": True, "show_full_dfs": show_full_dfs, "enable_cache": False},
        memory_size=10,
    )
    for df in dfs:
        logger.debug(f"df is: {str(df)}")
    response = panda_agent.chat(question)
    logger.debug(f"Response: {str(response)}")
    return response


async def verify_response(model, question, answer):
    verify_kwargs = {
        "question": question,
        "answer": answer,
    }
    prompt_engine = PromptEngine(model)
    system = prompt_engine.load_prompt("abilities/data/validation", **verify_kwargs)

    chat_parser = ChatParser(model)
    response = await chat_parser.get_parsed_response(
        system=system,
        containers=[Answer],
    )
    return response[0]


def preprocess_csv(raw_data, new_delimiter=","):
    possible_delimiters = ["\t", ";", "    "]  # Tab, semicolon, and four spaces

    # Check the first line (or more lines if necessary) to detect the delimiter.
    first_line = raw_data.split("\n")[0]
    detected_delimiter = None
    for d in possible_delimiters:
        if d in first_line:
            detected_delimiter = d
            break

    if detected_delimiter is None:
        return raw_data

    # Use the detected delimiter to parse and reformat the data.
    reader = csv.reader(raw_data.splitlines(), delimiter=detected_delimiter)

    # Creating a string buffer to export the CSV data
    output = StringIO()
    writer = csv.writer(output, delimiter=new_delimiter)

    for row in reader:
        writer.writerow(row)

    return output.getvalue()


async def preprocess_request(model: str, request: str, input: str) -> str:
    request_kwargs = {"request": request, "input": input}
    prompt_engine = PromptEngine(model)
    system = prompt_engine.load_prompt(
        "abilities/data/preprocess_request", **request_kwargs
    )

    chat_parser = ChatParser(model)
    response = await chat_parser.get_parsed_response(
        system=system,
        containers=[ShouldShowFullDfs],
    )
    return response[0].should_show_full_dfs
