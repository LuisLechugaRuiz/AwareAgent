from typing import Generic, List, TypeVar, Type, cast

from forge.helpers.parser.pydantic_parser import (
    get_json_schema,
    parse,
    ParseResult,
)
from forge.helpers.parser.loggable_base_model import LoggableBaseModel
from forge.helpers.parser.fix_format_prompt import (
    DEF_FIX_FORMAT_PROMPT,
)

from forge.sdk import (
    ForgeLogger,
    chat_completion_request
)

LOG = ForgeLogger(__name__)

T = TypeVar("T", bound=LoggableBaseModel)


class ChatParser(Generic[T]):
    def __init__(self, model):
        self.model = model

    @staticmethod
    def create_chat_message(role: str, content: str) -> dict[str, str]:
        return {"role": role, "content": content}

    async def get_response(self, system: str, user: str):
        messages = [
            self.create_chat_message("user", user),
            self.create_chat_message("system", system),
        ]
        response = await chat_completion_request(self.model, messages)
        return response["choices"][0]["message"]["content"]

    async def get_parsed_response(
        self,
        system: str,
        user: str,
        containers: List[Type[T]],
        retries: int = 2,
        fix_retries: int = 1,
    ) -> List[T]:
        output = []
        response = await self.get_response(system=system, user=user)
        for container in containers:
            success = False
            for _ in range(retries):
                parsed_response = await self.parse_response(
                    response, container, fix_retries=fix_retries
                )
                if parsed_response.result:
                    LOG.info(str(parsed_response.result))
                    parsed_response = cast(container, parsed_response.result)
                    output.append(parsed_response)
                    success = True
                    break
                else:
                    LOG.error(
                        "Couldn't parse/fix response, getting new response."
                    )
                    response = await self.get_response(
                        system=system, user=user
                    )
            if not success:
                LOG.critical(f"Failed to get a valid response after {retries} retries and {fix_retries} fix retries. Returning None...")
                output.append(None)
        return output

    async def parse_response(
        self, text: str, pydantic_object: Type[T], fix_retries=3
    ) -> ParseResult[T]:
        parsed_response = parse(text, pydantic_object)
        if parsed_response.result:
            return parsed_response
        else:
            error_msg = parsed_response.error_message
            LOG.error(
                f"Failing parsing object: {pydantic_object.__name__}, trying to fix autonomously...")
            # TODO: REMOVE ME LATER
            LOG.info("Response from the LLM: " + text + "error:" + error_msg)
        while fix_retries > 0:
            response_fix = await self.try_to_fix_format(text, error_msg, pydantic_object)
            if response_fix.result:
                LOG.info("Response format was fixed.")
                return response_fix
            fix_retries -= 1
            LOG.error(
                f"Couldn't fix format... remaining attempts to fix: {fix_retries}")
        return ParseResult(error_message=error_msg)

    async def try_to_fix_format(self, response, error_msg, pydantic_object):
        schema = get_json_schema([pydantic_object])
        fix_prompt = DEF_FIX_FORMAT_PROMPT.format(
            response=response,
            schema=schema,
            error_msg=error_msg,
        )
        fix_response = await self.get_response(
            system=fix_prompt,
            user="Please provide the correct format!",
        )
        result = parse(fix_response, pydantic_object)
        return result
