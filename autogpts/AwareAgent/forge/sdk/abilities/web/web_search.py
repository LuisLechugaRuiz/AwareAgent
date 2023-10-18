
from __future__ import annotations

import json
import os

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from duckduckgo_search import DDGS

from ..registry import ability

DUCKDUCKGO_MAX_ATTEMPTS = 3


@ability(
    name="get_relevant_links",
    description="Get most relevant links from Google search.",
    parameters=[
        {
            "name": "query",
            "description": "The search query",
            "type": "string",
            "required": True,
        }
    ],
    output_type="str",
)
async def get_relevant_links(agent, task_id: str, query: str) -> str:
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
