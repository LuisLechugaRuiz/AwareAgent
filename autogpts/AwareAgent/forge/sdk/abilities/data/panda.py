from forge.sdk.abilities.registry import ability
import pandas as pd
from pandasai.smart_dataframe import SmartDataframe
from pandasai.llm.openai import OpenAI


@ability(
    name="process_csv",
    description="Interact with and extract insights from structured data using natural language queries.",
    parameters=[
        {
            "name": "filepath",
            "description": "The path to the CSV file containing the structured data to analyze.",
            "type": "string",
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
async def conversational_data_insight(
    agent,
    task_id: str,
    filepath: str,
    question: str,
) -> str:
    """
    Analyze and interact with data from a CSV file using natural language.

    Args:
        filepath (str): The filepath to the CSV file containing the data.
        question (str): The natural language question to ask about the data.

    Returns:
        str: The insight or response generated from the data based on the question.
    """
    try:
        filepath = agent.workspace.base_path / task_id / filepath
        df = pd.read_csv(filepath)
        llm = OpenAI()
        llm.model = agent.model

        df = SmartDataframe(df, config={"llm": llm})
        response = df.chat(question)
        return response
    except Exception as e:
        return str(e)
