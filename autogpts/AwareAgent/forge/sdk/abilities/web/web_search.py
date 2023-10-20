
from __future__ import annotations

import json
import os
from pydantic import Field
from typing import List

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from forge.helpers.parser import ChatParser, LoggableBaseModel
from forge.sdk.abilities.registry import ability
from forge.sdk.abilities.web.web_selenium import read_webpage
from forge.sdk import ForgeLogger, PromptEngine

DUCKDUCKGO_MAX_ATTEMPTS = 3
LOG = ForgeLogger(__name__)


class Query(LoggableBaseModel):
    query: str = Field(
        description="The precise string of terms used for a Google search, crafted to target specific information or data."
    )


class SortedLinks(LoggableBaseModel):
    links: List[str] = Field(
        description="A list of handpicked URLs, ordered by relevance, with the most pertinent sources listed first."
    )


class Answer(LoggableBaseModel):
    reasoning: str = Field(
        description="A comprehensive explanation of the reasoning employed in formulating the answer, clearly demonstrating how it meets the validation criteria. Directly compare the result with each point of the validation standards, ensuring explicit alignment."
    )
    answer: str = Field(
        description="The concise information extracted from the source, directly addressing the query and relevant to the described objective."
    )
    partial_answer: bool = Field(
        description="Wether the response is a partial answer that should be saved."
    )
    validated: bool = Field(
        description="Indicates whether the current answer successfully meets the established validation constraints, ensuring it adequately responds to the initial description."
    )


@ability(
    name="web_search",
    description="Retrieves and confirms information from top-ranked Google search results based on specific inquiry criteria. Useful to find information on the web when you don't know the exact url. Be explicit about what do you want to find when using this ability.",
    parameters=[
        {
            "name": "description",
            "description": "A very detailed explanation of the information that needs to be retrieved. Be very explicit about it, include all the details.",
            "type": "string",
            "required": True,
        },
        {
            "name": "validation",
            "description": "A very detailed criteria assessing the accuracy and relevance of information obtained, ESSENTIAL for iterative refinement to ensure the highest quality result. Be very explicit about it, include all the details",
            "type": "string",
            "required": True,
        },
    ],
    output_type="str",
)
async def search_on_google(agent, task_id: str, description: str, validation: str) -> str:
    model = agent.model
    # Several stages.
    retries = 5  # TODO: Move to cfg
    max_search_links = 4  # TODO: Move to cfg
    results = []  # Tuple -> Query - Result.
    visited_links = []
    answer = ""
    while retries > 0:
        # 1. Create google query.
        previous_queries = None
        if results:
            previous_queries = results
        query_args = {
            "description": description,
            "validation": validation,
            "previous_queries": previous_queries,
        }
        query_response = await get_response(model, "abilities/web/query", query_args, Query)
        query = query_response.query
        LOG.info(f"New query: {query}")

        # 2. Get links from google.
        links = await get_relevant_links(query)

        # 3. Rank the links to find the most reliable sources
        rank_links_args = {
            "description": description,
            "validation": validation,
            "links": links,
            "visited_links": visited_links,
        }
        links_response = await get_response(model, "abilities/web/rank", rank_links_args, SortedLinks)
        links = links_response.links

        # 4. Scrape web (with summaries) and get answer, iterate until validation is met.
        max_search = min(max_search_links, len(links))
        for i in range(max_search):
            link = links[i]
            LOG.info(f"Visiting link: {link}")
            text = await read_webpage(agent=agent, task_id=task_id, url=link, question=query, get_links=False)
            visited_links.append(link)

            # 5. Validate text.
            validation_args = {
                "query": query,
                "text": text,
                "description": description,
                "validation": validation,
                "current_answer": answer,
            }
            validate_response = await get_response(model, "abilities/web/validation", validation_args, Answer)
            if validate_response.partial_answer:
                answer += validate_response.answer  # TODO: Verify if sum or substitution
            if validate_response.validated:
                answer = validate_response.answer
                return f"After scrapping the webs the answer is:\n{answer}\n\nPlease validate that this is the exact info that you require."
            LOG.info(f"Got non-validated answer: {answer}")

        retries -= 1
        results.append((query, answer))
    return f"Error while searching, after trying to get the information the non-validated answer is:\n{answer}"


async def get_response(model, prompt, system_kwargs, container):
    prompt_engine = PromptEngine(model)
    system = prompt_engine.load_prompt(prompt, **system_kwargs)

    chat_parser = ChatParser(model)
    response = await chat_parser.get_parsed_response(
        system=system,
        containers=[container],
    )
    return response[0]


async def get_relevant_links(query: str) -> str:
    """Return the results of a Google search using the official Google API

    Args:
        query (str): The search query.

    Returns:
        str: The results of the search.
    """

    try:
        # Get the Google API key and Custom Search Engine ID from the config file
        api_key = os.getenv("GOOGLE_API_KEY")
        custom_search_engine_id = os.getenv("GOOGLE_CUSTOM_SEARCH_ENGINE_ID")

        # Initialize the Custom Search API service
        service = build("customsearch", "v1", developerKey=api_key)

        # Send the search query and retrieve the results
        result = (
            service.cse()
            .list(q=query, cx=custom_search_engine_id, num=10)  # TODO: set by cfg
            .execute()
        )

        # Extract the search result items from the response
        search_results = result.get("items", [])

        # Create a list of only the URLs from the search results
        search_results_links = [item["link"] for item in search_results]

        return safe_google_results(search_results_links)

    except HttpError as e:
        # Handle errors in the API call
        error_details = json.loads(e.content.decode())

        # Check if the error is related to an invalid or missing API key
        if error_details.get("error", {}).get(
            "code"
        ) == 403 and "invalid API key" in error_details.get("error", {}).get(
            "message", ""
        ):
            return "The provided Google API key is invalid or missing."
        return f"Failed with exception {e}. Error: {str(error_details)}"


def safe_google_results(results: str | list) -> str:
    """
        Return the results of a Google search in a safe format.

    Args:
        results (str | list): The search results.

    Returns:
        str: The results of the search.
    """
    if isinstance(results, list):
        safe_message = json.dumps(
            [result.encode("utf-8", "ignore").decode("utf-8") for result in results]
        )
    else:
        safe_message = results.encode("utf-8", "ignore").decode("utf-8")
    return safe_message
