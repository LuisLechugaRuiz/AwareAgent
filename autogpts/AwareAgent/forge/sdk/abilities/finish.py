from forge.utils.logger.console_logger import ForgeLogger
from .registry import ability

logger = ForgeLogger(__name__)


@ability(
    name="finish",
    description="Use to shut down after completing goals or when facing unsolvable tasks.",
    parameters=[
        {
            "name": "reason",
            "description": "A summary to the user of how the goals were accomplished",
            "type": "string",
            "required": True,
        }
    ],
    output_type="None",
)
async def finish(
    agent,
    task_id: str,
    reason: str,
) -> str:
    """
    A function that takes in a string and exits the program

    Parameters:
        reason (str): A summary to the user of how the goals were accomplished.
    Returns:
        A result string from create chat completion. A list of suggestions to
            improve the code.
    """
    logger.info(reason, extra={"title": "Shutting down...\n"})
    return reason
