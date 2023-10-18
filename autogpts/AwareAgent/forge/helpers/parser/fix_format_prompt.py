DEF_FIX_FORMAT_PROMPT = """Your task is to correct the response: {response}.

Provide a answer with the correct format: {schema}.

We received the following error message: {error_msg}.

To complete your task, extract the content of the initial response and modify it to match the expected format.

Ensure that you remove any extra fields to strictly comply with the format instructions.

Identify fields with the same name and relevant information, then adapt the response to align with the desired output.

Eliminate duplicated data and provide only the pertinent context in the correct format.

Tips:
- If the response includes format instructions, remove them and return only the relevant response.
- If the response contains invalid answers such as null or None, replace them with an empty string."""
