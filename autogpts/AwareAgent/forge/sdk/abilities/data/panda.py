import pandas as pd
from pandasai.smart_dataframe import SmartDataframe
from pandasai.llm.openai import OpenAI

from forge.sdk.abilities.registry import ability
from forge.sdk import ForgeLogger

LOG = ForgeLogger(__name__)


@ability(
    name="proccess_csv_and_save",
    description="Proccess a CSV and save the result on another file using natural language queries.",
    parameters=[
        {
            "name": "input_filename",
            "description": "The name of the CSV file containing the structured data to analyze.",
            "type": "string",
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
            "description": "The request that will be used to proccess the data and save the result.",
            "type": "string",
            "required": True,
        },
    ],
    output_type="str",
)
async def proccess_csv_and_save(
    agent, task_id: str, input_filename: str, output_filename: str, request: str
) -> str:
    attempts = 2
    initial_request = request
    while attempts > 0:
        response = await pandasai_chatbot(
            agent, task_id, input_filename, request, "dataframe"
        )
        # Check if the response is a DataFrame before saving
        if isinstance(response, SmartDataframe):
            dataframe = response.original_import()
            # Construct the file path
            output_filepath = agent.workspace.base_path / task_id / output_filename

            # Save DataFrame to CSV
            dataframe.to_csv(output_filepath, index=False)
            await agent.db.create_artifact(
                task_id=task_id,
                file_name=output_filename,
                relative_path="",
                agent_created=True,
            )
            return f"Dataframe was saved successfully to {output_filename}"
    else:
        request = f"{initial_request}\n The previous response failed due to: {str(response)}, please try again."
        attempts -= 1

    return f"Failed with error: {str(response)}"


@ability(
    name="get_insights_from_csv",
    description="Extract insights from a CSV using natural language queries.",
    parameters=[
        {
            "name": "filename",
            "description": "The name of the CSV file containing the structured data to analyze.",
            "type": "string",
            "required": True,
        },
        {
            "name": "question",
            "description": "A question referring to the data's content, which the system will interpret and respond to with the appropriate data insights.",
            "type": "string",
            "required": True,
        },
        {
            "name": "output_format",
            "description": "The expected output format. Only 'number' and 'string' are supported!",
            "type": "string",
            "required": True,
        },
    ],
    output_type="str",
)
async def get_insights_from_csv(
    agent, task_id: str, filename: str, question: str, output_format: str
) -> str:
    if output_format not in ["number", "string"]:
        output_format = "string"
    response_str = await pandasai_chatbot(
        agent, task_id, filename, question, output_format
    )
    return response_str


async def pandasai_chatbot(
    agent, task_id: str, filepath: str, question: str, output_type: str
) -> str:
    attempts = 2
    initial_question = question
    error = ""
    while attempts > 0:
        try:
            full_filepath = agent.workspace.base_path / task_id / filepath
            df = pd.read_csv(full_filepath)
            llm = OpenAI()
            llm.model = agent.model

            df = SmartDataframe(df, config={"llm": llm})
            response = df.chat(question, output_type)
            LOG.debug(f"Response: + {str(response)}")
            return response
        except Exception as e:
            question = f"{initial_question} \n The previous response failed due to: {str(e)}, please try again."
            attempts -= 1
            error = e
    return f"An error occurred: {str(error)}"
