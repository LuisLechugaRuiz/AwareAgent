from forge.helpers.parser import ChatParser
from forge.sdk.abilities.registry import ability


@ability(
    name="get_GPT4_help",
    description="Prompt GPT-4 to help you accomplish your goals. Useful when you need to perform intermediate steps. Keep in mind that the only info available to GPT-4 will be the one added on this prompt, so please provide a very detailed description of the problem.",
    parameters=[
        {
            "name": "prompt",
            "description": "The prompt to send to GPT-4. This will be all the info sent to GPT-4 so please provide the info that you need. He doesn't have access to your short term memory, in case you need him to process some info from there, please add it to the prompt.",
            "type": "string",
            "required": True,
        },
    ],
    output_type="str",
)
async def get_GPT4_help(
    agent,
    task_id: str,
    prompt: str,
) -> str:
    """
    Prompt GPT-4 to help you accomplish your goals. Useful when you need to perform intermediate steps

    Args:
        prompt (str): The prompt to send to GPT-4.

    Returns:
        str: The answer from GPT-4.
    """
    chat_parser = ChatParser(agent.model)
    answer = await chat_parser.get_response(
        system="Provide a final answer to the user request.",
        user=prompt,
    )
    return answer
