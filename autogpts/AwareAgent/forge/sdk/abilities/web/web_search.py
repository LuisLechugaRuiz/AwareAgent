from __future__ import annotations

import asyncio
from googleapiclient.errors import HttpError
import json
import os
from pydantic import Field
from typing import List
import urllib.parse

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from forge.agent.agent import ForgeAgent
from forge.helpers.parser import ChatParser, LoggableBaseModel
from forge.helpers.parser.pydantic_parser import get_json_schema
from forge.sdk.abilities.registry import ability
from forge.sdk.abilities.web.web_selenium import read_webpage_and_summarize
from forge.sdk.config.config import Config
from forge.sdk import PromptEngine
from forge.utils.logger.console_logger import ForgeLogger
from forge.utils.process_tokens import count_string_tokens, preprocess_text
from forge.sdk.memory.long_term_memory.weaviate import WeaviateMemory

LOG = ForgeLogger(__name__)


class Answer(LoggableBaseModel):
    data_selection_reasoning: str = Field(
        description="Compare all sourced information, ultimately selecting data that best aligns with and adheres to the validation criteria and target info.",
    )
    answer: str = Field(
        description="The information, extracted with utmost precision, must be the absolute, unrounded figures or data, as specified within the source. Generalizations, rounded figures, or any form of approximation are unacceptable. The response must mirror the exact information from the source, satisfying the validation criteria in an unequivocal manner.",
    )
    validation_reasoning: str = Field(
        description="Break down the objective and the validation criteria into their most fundamental components, conducting a detailed, point-by-point assessment. For each specific criterion, clearly denote whether the answer 'succeeds' or 'fails' to meet the standard (e.g., 'Criterion A: True', 'Criterion B: False')."
    )
    criteria_met: bool = Field(
        description="Flag to indicate if the answer satisfy the validation criteria in an unequivocal manner. If the validation criteria are not met, the response should be False.",
    )


@ability(
    name="search_on_google",
    description="Retrieves relevant data from top Google search results.",
    parameters=[
        {
            "name": "queries",
            "description": "The queries used to search on google. Be creative and use different queries to get the best results. This field is mandatory!",
            "type": "List[str]",
            "required": True,
        },
        {
            "name": "validation",
            "description": "Clear criteria that the response must satisfy, such as data format, exact figures, or key details that are non-negotiable for acceptance.",
            "type": "string",
            "required": True,
        },
    ],
    output_type="str",
)
async def search_on_google(
    agent: ForgeAgent,
    task_id: str,
    queries: List[str],
    validation: str,
) -> str:
    memory = agent.get_long_term_memory()
    return await _search_on_google(memory, queries, validation)


async def _search_on_google(
    memory: WeaviateMemory,
    queries: List[str],
    validation: str,
) -> str:
    if len(queries) == 0:
        return "No queries provided. Please provide at least one query."

    model = Config().smart_llm_model
    memory.reset_web_data()  # Reset web data to avoid saving data from previous runs, it can get the same info again and again... TODO: Fix it.

    # 1. Get all the top results from google.
    urls = []
    unique_urls = set(urls)  # Convert the initial list to a set for fast lookup
    links_per_query = Config().max_search_links // len(queries)
    extra_links = Config().max_search_links % len(queries)

    for i, query in enumerate(queries):
        query += " -filetype:pdf"  # Exclude pdfs
        new_urls = await get_urls(query)
        LOG.debug(f"New urls: {new_urls}")

        max_query_links = links_per_query + (1 if i < extra_links else 0)
        added_count = 0  # Counter for new URLs added

        for url in new_urls:
            # If we've reached the maximum number of new links to add, break from the loop
            if added_count >= max_query_links:
                break

            # Only add URLs which are not already in the unique_urls set
            if url not in unique_urls:
                unique_urls.add(url)
                urls.append(url)
                added_count += 1

    # Get worst case scenario for the prompt.
    chunk_max_tokens = get_chunk_max_tokens(
        model, urls, queries, validation
    )
    LOG.info(f"Chunk max tokens: {chunk_max_tokens}")
    for url in urls:
        LOG.info(f"- Saving url: {url}")
        # 1.1 read raw web page.
        text = await read_webpage_and_summarize(
            model=Config().fast_llm_model,
            url=url,
            question=query,
            get_links=False,
            should_summarize=False,
        )
        # 1.2 split into chunks.
        chunks = await get_chunks(
            text,
            model=model,
            chunk_max_tokens=chunk_max_tokens,
        )
        for chunk in chunks:
            # 1.3 save chunks into memory.
            memory.store_web_data(url=url, query=query, content=chunk)

    # 2. Get the most relevant data. TODO -> decide which query to search
    top_results = memory.search_web_data(
        query=queries[0], num_relevant=Config().web_search_top_results
    )
    if not top_results:
        return "No results found..."

    # 3. Get answer and validate it.
    validation_args = {
        "sources": top_results,
        "validation": validation,
    }
    validate_response = await get_response(
        model, "abilities/web/validation", validation_args, Answer
    )
    answer = validate_response.answer
    if validate_response.criteria_met:
        return f"After scrapping the webs the answer is:\n{answer}\nPlease verify the response and the format!"
    return f"Couldn't validate the response. After trying to get the information the non-validated answer is:\n{answer}"


def get_longest_source(urls, queries):
    def get_max_length(items):
        max_item = ""
        max_length = 0
        for item in items:
            if len(item) > max_length:
                max_length = len(item)
                max_item = item
        return max_item

    max_length_url = get_max_length(urls)
    max_length_query = get_max_length(queries)
    max_source_dict = {"url": max_length_url, "query": max_length_query, "content": ""}
    return max_source_dict


def get_chunk_max_tokens(model, urls, queries, validation):
    # Split text into chunks in a way that ensures that later we can use web_search_top_results chunks on a single prompt.
    # We should be able to use validate prompt and get an answer of max tokens: web_answer_reserved_tokens
    max_tokens = (
        Config().get_max_model_tokens(model) - Config().web_answer_reserved_tokens
    )
    chunk_max_tokens = max_tokens / Config().web_search_top_results
    LOG.debug(f"Max tokens: {max_tokens}")

    # Get longest possible source
    max_source = get_longest_source(urls, queries)
    sources = [max_source for _ in range(Config().web_search_top_results)]
    validation_args = {
        "sources": sources,
        "validation": validation,
    }
    # Get worst case scenario for the prompt.
    prompt_engine = PromptEngine(model)
    raw_system_prompt = prompt_engine.load_prompt(
        "abilities/web/validation", **validation_args
    )
    user_kwargs = {"schema": get_json_schema([Answer])}
    raw_user_prompt = prompt_engine.load_prompt("new/user", **user_kwargs)
    raw_prompt = raw_system_prompt + raw_user_prompt
    prompt_tokens = count_string_tokens(raw_prompt, model)

    # Distribute the tokens among the chunks
    chunk_max_tokens -= prompt_tokens / Config().web_search_top_results
    return chunk_max_tokens


async def get_chunks(
    text: str,
    model: str,
    chunk_max_tokens: int,
) -> List[str]:
    chunks = preprocess_text("", text, chunk_max_tokens=chunk_max_tokens, model=model)
    return chunks


async def get_response(model, prompt, system_kwargs, container):
    prompt_engine = PromptEngine(model)
    system = prompt_engine.load_prompt(prompt, **system_kwargs)
    system_tokens = count_string_tokens(system, model)
    LOG.debug(f"System tokens: {system_tokens}")
    LOG.debug(f"System prompt: {system}")

    chat_parser = ChatParser(model)
    response = await chat_parser.get_parsed_response(
        system=system,
        containers=[container],
    )
    return response[0]


async def get_urls(query: str, max_retries: int = 3, delay: int = 5) -> List[str]:
    """Return the results of a Google search using the official Google API, with retries in case of failure.

    Args:
        query (str): The search query.
        max_retries (int): The maximum number of retries if the request fails. Defaults to 3.
        delay (int): The number of seconds to wait between retries. Defaults to 5.

    Returns:
        List[str]: List of URLs from the search results.
    """

    for attempt in range(max_retries):
        try:
            # Get the Google API key and Custom Search Engine ID from the config file
            api_key = os.getenv("GOOGLE_API_KEY")
            custom_search_engine_id = os.getenv("GOOGLE_CUSTOM_SEARCH_ENGINE_ID")

            # Initialize the Custom Search API service
            service = build("customsearch", "v1", developerKey=api_key)

            # Send the search query and retrieve the results
            result = service.cse().list(q=query, cx=custom_search_engine_id, num=10).execute()

            # Extract the search result items from the response
            search_results = result.get("items", [])

            # Create a list of only the URLs from the search results
            return [item.get("link") for item in search_results if item.get("link") is not None]

        except HttpError as e:
            if attempt < max_retries - 1:  # Check if more retries are left
                await asyncio.sleep(delay)  # Asynchronous sleep before retrying
                continue
            else:
                error_details = e.content.decode() if e.content else str(e)
                return f"Failed after {max_retries} attempts. Error: {error_details}"
        except Exception as e:
            if attempt < max_retries - 1:  # Check if more retries are left
                await asyncio.sleep(delay)  # Asynchronous sleep before retrying
                continue
            else:
                return f"An unexpected error occurred: {str(e)}"

    return []


async def main():
    target_info = "Tesla exact revenue 2022"
    queries = ["Tesla revenue 2022", "Tesla exact revenue 2022 million dollar"]
    validation = "Result uses the US notation, with a precision rounded to the nearest million dollars"
    memory = WeaviateMemory()
    return await _search_on_google(
        memory, queries, validation=validation
    )


if __name__ == "__main__":
    print(main())
