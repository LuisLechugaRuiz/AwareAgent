from ..registry import ability


@ability(
    name="write_file",
    description="Create a file (if it doesn't exist) and write data to it.",
    parameters=[
        {
            "name": "filename",
            "description": "Name of the file to create. Don't forget to include the extension.",
            "type": "string",
            "required": True,
        },
        {
            "name": "text",
            "description": "Text to write to the file. Verify the format before writing it!",
            "type": "string",
            "required": True,
        },
    ],
    output_type="None",
)
async def write_file(agent, task_id: str, filename: str, text: str):
    """
    Write data to a file
    """
    data = text.encode("utf-8")

    agent.workspace.write(task_id=task_id, path=filename, data=data)
    try:
        await agent.db.create_artifact(
            task_id=task_id,
            file_name=filename,
            relative_path="",
            agent_created=True,
        )
        return f"Successfully created file: {filename} and added the following text: {data.decode('utf-8')}"
    except Exception as e:
        return f"Failing to create file: {filename} in the database. Error: {e}"


@ability(
    name="read_file",
    description="Read data from a file",
    parameters=[
        {
            "name": "file_path",
            "description": "Path to the file",
            "type": "string",
            "required": True,
        },
    ],
    output_type="bytes",
)
async def read_file(agent, task_id: str, file_path: str) -> bytes:
    """
    Read data from a file
    """
    return agent.workspace.read(task_id=task_id, path=file_path).decode("utf-8")
