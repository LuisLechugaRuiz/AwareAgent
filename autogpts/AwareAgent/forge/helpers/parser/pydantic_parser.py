import json
import re
from typing import cast, List, Generic, Optional, Type, TypeVar

from pydantic import BaseModel, ValidationError

PYDANTIC_FORMAT_INSTRUCTIONS = """Please provide a JSON object that conforms to the JSON schema provided below.

Example of output schema: {{"Example": {{"properties": {{"foo": {{"title": "Foo", "description": "a list of strings", "type": "array", "items": {{"type": "string"}}}}}}, "required": ["foo"]}}}}
Well-formatted instance of the schema: {{"Example": {{"foo": ["bar", "baz"]}}.
Not well-formatted instance {{"Example": {{"properties": {{"foo": ["bar", "baz"]}}}}.

Here is the output schema:
```
{schema}
```"""


def parse_pydantic_object(text: str, pydantic_object: Type[BaseModel]):
    def preprocess(text: str) -> str:
        text = text.replace("True", "true").replace("False", "false")
        return text

    try:
        text = preprocess(text)
        pattern = rf"{pydantic_object.__name__}\s*:?\s*"
        match = re.search(pattern, text)

        if not match:
            raise ValueError(f"{pydantic_object.__name__} not found in text")

        start = match.end()

        # Initialize the JSONDecoder
        decoder = json.JSONDecoder()

        # Find the next valid JSON object
        while start < len(text):
            try:
                json_obj, _ = decoder.raw_decode(text, start)
                # Check if the JSON object has the required keys
                if all(
                    key in json_obj for key in pydantic_object.schema()["properties"]
                ):
                    break
                else:
                    # Update the start position to continue searching
                    start += 1
            except json.JSONDecodeError:
                start += 1

        # Parse the JSON object and return the pydantic_object
        return pydantic_object.parse_obj(json_obj)

    except (ValueError, json.JSONDecodeError, ValidationError) as e:
        name = pydantic_object.__name__
        msg = f"Failed to parse {name} from completion {text}. Got: {e}"
        raise Exception(msg)


def get_json_schema(objects: List[Type[BaseModel]]) -> str:
    combined_schema = {}

    for obj in objects:
        schema = obj.schema()

        # Remove extraneous fields.
        reduced_schema = schema
        if "title" in reduced_schema:
            del reduced_schema["title"]
        if "type" in reduced_schema:
            del reduced_schema["type"]

        # Add object name to the schema
        object_name = obj.__name__
        combined_schema[object_name] = reduced_schema

    # Ensure json in context is well-formed with double quotes.
    json_schema = json.dumps(combined_schema)

    return json_schema


T = TypeVar("T", bound=BaseModel)


class ParseResult(Generic[T]):
    def __init__(
        self,
        result: Optional[T] = None,
        error_message: Optional[str] = None,
    ):
        self.result = result
        self.error_message = error_message


def parse(text: str, pydantic_object: Type[T]) -> ParseResult[T]:
    try:
        result = parse_pydantic_object(text, pydantic_object)
        return ParseResult(result=cast(T, result))
    except Exception as e:
        return ParseResult(error_message=e)
