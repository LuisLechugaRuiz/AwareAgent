import abc
import os

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Singleton(abc.ABCMeta, type):
    """
    Singleton metaclass for ensuring only one instance of a class.
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class AbstractSingleton(abc.ABC, metaclass=Singleton):
    pass


class Config(metaclass=Singleton):
    """
    Configuration class to store the state of bools for different scripts access.
    """

    def __init__(self):
        # KEYS
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.custom_search_engine_id = os.getenv("CUSTOM_SEARCH_ENGINE_ID")
        self.max_goal_iterations = int(os.getenv("MAX_GOAL_ITERATIONS", "10"))
        # CONFIG
        self.debug_mode = os.getenv("DEBUG_MODE", "False") == "True"
        self.fast_llm_model = os.getenv("FAST_LLM_MODEL", "gpt-3.5-turbo")
        self.fast_token_limit = int(os.getenv("FAST_TOKEN_LIMIT", 4000))
        self.smart_llm_model = os.getenv(
            "SMART_LLM_MODEL", "gpt-4"
        )
        self.smart_token_limit = int(os.getenv("SMART_TOKEN_LIMIT", 8000))
        # MEMORY MANAGEMENT
        self.episodes_tokens_percentage = float(
            os.getenv("EPISODES_TOKENS_PERCENTAGE", 0.4)
        )
        self.summary_tokens_percentage = float(
            os.getenv("SUMMARY_TOKENS_PERCENTAGE", 0.1)
        )
        self.episodes_overlap_tokens = int(os.getenv("EPISODES_OVERLAP_TOKENS", 100))

        self.browse_spacy_language_model = os.getenv(
            "BROWSE_SPACY_LANGUAGE_MODEL", "en_core_web_sm"
        )
        # self.elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")

        # User agent headers to use when browsing web
        # Some websites might just completely deny request with an error code if no user agent was found.
        self.user_agent_header = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36"
        }
        self.memory_index = os.getenv("MEMORY_INDEX", "aware")
        self.long_term_memory_backend = os.getenv(
            "LONG_TERM_MEMORY_BACKEND", "weaviate"
        )
        # Weaviate
        self.weaviate_url = os.getenv("WEAVIATE_URL", "http://weaviate")
        self.local_weaviate_url = os.getenv(
            "LOCAL_WEAVIATE_URL", "http://localhost"
        )  # TODO: Remove after moving to cloud
        self.weaviate_port = os.getenv("WEAVIATE_PORT", "9090")
        self.weaviate_key = os.getenv("WEAVIATE_KEY")

        # Search
        self.max_search_links = os.getenv("MAX_SEARCH_LINKS", 3)  # Max links to search, will be distributed between all the queries
        self.web_search_top_results = os.getenv("WEB_SEARCH_TOP_RESULTS", 3)
        self.web_answer_reserved_tokens = os.getenv("WEB_ANSWER_RESERVED_TOKENS", 6000)

        # Coding
        self.self_improvement = os.getenv("SELF_IMPROVEMENT", "False") == "True"
        self.max_improvement_retries = os.getenv("MAX_IMPROVEMENT_RETRIES", 3)

    def get_max_model_tokens(self, model) -> int:
        """Get the tokens count of the model"""

        if model == Config().fast_llm_model:
            return Config().fast_token_limit
        else:
            return Config().smart_token_limit
