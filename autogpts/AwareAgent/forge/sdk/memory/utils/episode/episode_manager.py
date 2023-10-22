from typing import Any, List, Optional
import spacy
import subprocess
from spacy.util import get_package_path
from pydantic import Field

from forge.sdk.memory.long_term_memory.weaviate import WeaviateMemory
from forge.sdk.memory.utils.episode.episode import Episode
from forge.sdk import PromptEngine

from forge.helpers.parser import ChatParser, LoggableBaseModel
from forge.utils.process_tokens import count_string_tokens
from forge.sdk.config.config import Config

from forge.utils.logger.file_logger import FileLogger


class EpisodeManager(object):
    def __init__(
        self,
        model: str = Config().fast_llm_model,
        episodes_tokens_percentage: float = Config().episodes_tokens_percentage,
        summary_tokens_percentage: float = Config().episodes_tokens_percentage,
    ):
        self.long_term_memory = WeaviateMemory()
        self.logger = FileLogger("episode_manager")
        self.episodes: List[Episode] = []
        self.summary: Optional[Episode] = None
        self.episodes_max_tokens = self.calculate_max_tokens(
            model=model, tokens_percentage=episodes_tokens_percentage
        )
        self.summary_max_tokens = self.calculate_max_tokens(
            model=model, tokens_percentage=summary_tokens_percentage
        )
        self.model = model

    def calculate_max_tokens(self, model: str, tokens_percentage: float) -> int:
        """Calculate the max tokens allowed"""

        if model == Config().fast_llm_model:
            return round(tokens_percentage * Config().fast_token_limit)
        else:
            return round(tokens_percentage * Config().smart_token_limit)

    async def _add_episode(self, new_episode: Episode) -> None:
        """Add a new episode"""

        new_episode_tokens = self.get_tokens_count(new_episode.get_description())
        current_tokens = sum(
            self.get_tokens_count(episode.get_description())
            for episode in self.episodes
        )

        excess_tokens = (current_tokens + new_episode_tokens) - self.episodes_max_tokens
        if excess_tokens > 0:
            # We need to remove some parts of episodes
            excess_episodes: List[Episode] = []
            tokens_removed = 0

            while self.episodes and tokens_removed < excess_tokens:
                # Start with the oldest episodes
                current_episode = self.episodes[0]
                episode_tokens = self.get_tokens_count(
                    current_episode.get_description()
                )

                if tokens_removed + episode_tokens <= excess_tokens:
                    # If the whole episode can be removed within the limit
                    tokens_removed += episode_tokens
                    excess_episodes.append(current_episode)
                    self.episodes.pop(0)
                else:
                    break
            if excess_episodes:
                if self.summary:
                    # Insert current summary as the initial episode
                    excess_episodes.insert(0, self.summary)
                # Create a meta-episode summarizing the episodes.
                self.summary = await self.create_meta_episode(
                    excess_episodes, max_tokens=self.summary_max_tokens
                )
        self.episodes.append(new_episode)

    def clear_episodes(self) -> None:
        """Clear all the episodes"""

        self.episodes = []

    async def create_episodes(
        self,
        goal: str,
        capability: str,
        ability: str,
        arguments: str,
        observation: str,
    ) -> None:
        """Create a list of new episode based on the given operations"""

        max_overview_tokens = round(self.episodes_max_tokens * 0.2)
        overview_raw_prompt = self.get_overview_prompt(
            task_description=goal,
            action=ability,
            observation="",
            max_overview_tokens=self.episodes_max_tokens,
        )
        # Compress the text into chunks with the same max tokens than an episode.
        chunk_max_tokens = (
            Config().get_max_model_tokens(self.model) - max_overview_tokens - 100
        )
        self.logger.info(f"chunk_max_tokens: {chunk_max_tokens}", should_print=False)
        full_text = Episode.get_format(goal, ability, arguments, observation)
        prefix = """This is the {n_episode} chunk of a sequence of {total_episodes} chunks while performing action: {action}\n"""
        chunks = self.preprocess_text(
            raw_prompt=overview_raw_prompt,
            text=full_text,
            chunk_max_tokens=chunk_max_tokens,
            prefix=prefix,
        )
        episodes = []
        if len(chunks) == 0:
            return
        if len(chunks) == 1:
            episode = await self.create_episode(
                goal=goal,
                capability=capability,
                ability=ability,
                arguments=arguments,
                observation=observation,
            )
            await self._add_episode(new_episode=episode)
            return
        for idx, chunk in enumerate(chunks):
            prefix_formatted = prefix.format(
                n_episode=idx, total_episodes=len(chunks), action=ability
            )
            chunk = prefix_formatted + chunk
            # Create all the episodes and save them on long term memory.
            episodes.append(
                await self.create_episode(
                    goal=goal,
                    capability=capability,
                    ability=ability,
                    arguments=arguments,
                    observation=chunk,
                )
            )
        episode = await self.create_meta_episode(episodes)  # Create a meta episode.
        await self._add_episode(new_episode=episode)

    async def create_episode(
        self,
        goal,
        capability,
        ability,
        arguments,
        observation,
    ) -> Optional[Episode]:
        max_overview_tokens = round(self.episodes_max_tokens * 0.2)
        overview_prompt = self.get_overview_prompt(
            task_description=goal,
            action=ability,
            observation=observation,
            max_overview_tokens=max_overview_tokens,
        )
        self.logger.info(
            f"Creating episode from summarized observation using prompt: {overview_prompt}",
            should_print=False,
        )
        # overview = await ChatParser(model=self.model).get_response(
        #    system=overview_prompt,
        #    user="Please include only the information specifically requested, without any further additions."
        # )
        overview = "No implemented yet!"  # TODO: Implement after hackathon to store/retrive episodes from long term memory.
        episode = Episode(
            content=observation,
            overview=overview,
        )

        episode.add_execution(
            goal=goal,
            capability=capability,
            ability=ability,
            arguments=arguments,
            observation=observation,
        )
        episode.set_order(order=len(self.episodes))
        self.save_on_long_term_memory(
            episode=episode
        )  # All episodes are saved on long term memory
        return episode

    async def create_meta_episode(
        self,
        episodes: List[Episode],
        question: Optional[str] = None,
        max_tokens: int = None,
    ) -> Optional[Episode]:
        """Create and save a new meta episode based on the given episodes"""

        if len(episodes) == 0:
            return None
        if len(episodes) == 1:
            return episodes[0]

        raw_prompt = self.get_summary_prompt(
            new_content="", previous_content="", max_tokens=max_tokens, question=question
        )
        raw_prompt_tokens = count_string_tokens(string=raw_prompt)
        if max_tokens is None:
            max_tokens = Config().get_max_model_tokens(self.model)

        i = 0
        num_episodes = len(episodes)
        chunk_max_tokens = max_tokens - raw_prompt_tokens
        current_chunk = ""
        meta_episode = None
        for idx, episode in enumerate(episodes):
            future_chunk = current_chunk + episode.get_description()
            if (
                count_string_tokens(string=future_chunk) >= chunk_max_tokens
                or idx == num_episodes - 1
            ):
                # TODO: Make this better.
                if idx == num_episodes - 1:
                    current_chunk = future_chunk
                self.logger.info(f"Creating meta episode {i}", should_print=True)
                if meta_episode:
                    meta_episode_str = meta_episode.get_description()
                else:
                    meta_episode_str = ""
                prompt = self.get_summary_prompt(
                    new_content=current_chunk,
                    previous_content=meta_episode_str,
                    max_tokens=max_tokens,
                    question=question,
                )

                # parsed_response = await ChatParser(
                #    model=self.model
                #).get_parsed_response(
                #    system=prompt,
                #    containers=[Summary],
                #)
                # TODO: SUMMARY SHOULD NOT BE ANOTHER EPISODE. CREATE SUMMARY OBJECT ON WEAVIATE.

                parsed_response = None  # DISABLE FOR NOW AS LONG TERM MEMORY IS NOT USED FOR HACKATHON.
                if parsed_response:
                    meta_episode = Episode(
                        overview=parsed_response[0].overview,
                    )
                    meta_episode._observation = parsed_response[0].content
                else:
                    meta_episode = Episode(
                        overview=f"New summary integrating multiples episodes on {episode.overview}, failed to parse.",
                    )  # Save raw data.
                    meta_episode._observation = current_chunk
                # Update raw_prompt_tokens to do not exceed the chunk_max_tokens
                raw_prompt = self.get_summary_prompt(
                    new_content="",
                    previous_content=meta_episode.get_description(),
                    max_tokens=max_tokens,
                    question=question,
                )
                raw_prompt_tokens = (
                    count_string_tokens(string=raw_prompt) + 100
                )  # Some extra tokens just in case.
                chunk_max_tokens = max_tokens - raw_prompt_tokens
                current_chunk = ""
                i = i + 1
            else:
                current_chunk = future_chunk
        episodes_uuid = []
        for episode in episodes:
            episode_uuid = episode.get_uuid()
            if episode_uuid:
                episodes_uuid.append(episode_uuid)
        if meta_episode:
            self.save_on_long_term_memory(
                episode=meta_episode,
                child_episodes_uuid=episodes_uuid,
            )
            meta_episode.set_order(order=0)
            meta_episode.add_child_episodes(episodes=episodes)
            return meta_episode
        return None

    def clear(self) -> None:
        """Clear current episodes"""

        self.episodes = []

    def get_episodes(self) -> List[Episode]:
        """Get episodes"""

        return self.episodes

    def get_episodes_str(self) -> Optional[str]:
        """Get the current summary in reverse order of the episodes."""
        if self.episodes:
            # Reverse the list of episodes
            reversed_episodes = self.episodes[::-1]

            # Join the episode descriptions, now in reverse order
            return "\n\n".join(
                [episode.get_description() for episode in reversed_episodes]
            )
        return None

    def get_summary_str(self) -> Optional[str]:
        """Get the current summary"""

        return self.summary.overview if self.summary else None

    def get_summary_prompt(
        self,
        new_content: str,
        previous_content: str,
        max_tokens: int,
        question: Optional[str] = None,
    ) -> str:
        """Get the meta episode prompt"""
        meta_episode_kwargs = {
            "new_content": new_content,
            "previous_content": previous_content,
            "question": question,
            "max_tokens": max_tokens,
        }
        prompt_engine = PromptEngine(self.model)
        return prompt_engine.load_prompt(
            "memory/utils/episode/meta_episode", **meta_episode_kwargs
        )

    def get_overview_prompt(
        self,
        task_description: str,
        action: str,
        observation: str,
        max_overview_tokens: int,
    ):
        overview_prompt_kwargs = {
            "task_description": task_description,
            "action": action,
            "observation": observation,
            "max_overview_tokens": max_overview_tokens,
        }
        prompt_engine = PromptEngine(self.model)
        return prompt_engine.load_prompt(
            "memory/utils/episode/overview", **overview_prompt_kwargs
        )

    def get_tokens_count(self, episodes_str) -> int:
        """Get the tokens count of the summaries"""

        return count_string_tokens(episodes_str)

    def to_dict(self) -> dict[str, Any]:
        """Buffer to dict"""

        return {
            "model": self.model,
            "summary": self.summary.to_dict() if self.summary else None,
            "episodes": [episode.to_dict() for episode in self.episodes],
        }

    @classmethod
    def from_dict(cls, data):
        episode_manager = cls(
            model=data["model"],
        )
        if data["summary"]:
            episode_manager.summary = Episode.from_dict(data=data["summary"])
        for episode_data in data["episodes"]:
            episode = Episode.from_dict(data=episode_data)
            episode_manager.episodes.append(episode)
        return episode_manager

    def save_on_long_term_memory(
        self,
        episode: Episode,
        child_episodes_uuid: List[str] = [],
    ) -> None:
        """Add episode to long term memory"""

        self.logger.info(
            f"Adding episode: {episode.get_description()}", should_print=False
        )
        episode_uuid = self.long_term_memory.store_episode(
            overview=episode.overview,
            goal=episode._goal,
            capability=episode._capability,
            ability=episode._ability,
            arguments=episode._arguments,
            observation=episode._observation,
            created_at=episode._creation_time,
            child_episodes_uuid=child_episodes_uuid,
        )
        episode.link_to_uuid(uuid=episode_uuid)

    def preprocess_text(
        self,
        raw_prompt: str,
        text: str,
        chunk_max_tokens: int,
        prefix: Optional[str] = None,
    ) -> List[str]:
        """Preprocess text"""

        model_name = Config().browse_spacy_language_model
        try:
            model_path = get_package_path(model_name)
        except Exception:
            model_path = None

        if model_path is None:
            # Install the model if it's not available
            print(f"{model_name} is not installed. Installing now...")
            subprocess.check_call(["python", "-m", "spacy", "download", model_name])
        try:
            nlp = spacy.load(model_name)
        except Exception:
            raise Exception(f"Failed to load the spacy model: {model_name}")
        nlp.add_pipe("sentencizer")
        doc = nlp(text)
        sentences = [sent.text for sent in doc.sents]

        sentences_length = 0
        for sentence in sentences:
            sentences_length += count_string_tokens(sentence)
        self.logger.info(f"Total tokens: {sentences_length}", should_print=False)
        prompt_tokens = count_string_tokens(string=raw_prompt, model_name=self.model)
        if prefix:
            prompt_tokens = prompt_tokens + count_string_tokens(
                string=prefix, model_name=self.model
            )
        chunk_max_tokens = chunk_max_tokens - prompt_tokens

        chunks = [sentences.pop(0)]
        for sentence in sentences:
            # Accumulate on current chunk until max tokens is reached.
            future_chunk = chunks[-1] + " " + sentence
            if count_string_tokens(string=future_chunk) >= chunk_max_tokens:
                overlap_text = self.get_tokens(
                    text=chunks[-1], max_tokens=Config().episodes_overlap_tokens
                )
                chunks.append(overlap_text + " " + sentence)
            else:
                chunks[-1] = future_chunk
        self.logger.info(f"Number of chunks: {len(chunks)}", should_print=False)
        return chunks

    def get_tokens(self, text: str, max_tokens: int):
        """Get words from text"""
        words = text.split()
        tokens = ""
        for word in words:
            future_tokens = tokens + word + " "
            if count_string_tokens(string=future_tokens) >= max_tokens:
                return tokens.strip()
            tokens = future_tokens
        return tokens.strip()


class Summary(LoggableBaseModel):
    overview: str = Field(
        description="A succinct encapsulation of the main points and themes derived from the detailed content, crafted to provide a quick and coherent understanding without delving into the intricacies. This field prioritizes brevity and clarity, offering a clear snapshot of the larger narrative.",
    )
    content: str = Field(
        description="An elaborate compilation of all the essential information, including nuanced details, relevant citations, and critical analyses. This field is designed for a deep dive into the subject matter, encompassing all relevant aspects to present a comprehensive and well-rounded view of the topic. It's the cornerstone for informed decision-making, academic insights, or thorough understanding, depending on the context.",
    )
