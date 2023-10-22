from typing import List, Optional
import weaviate

from forge.sdk.memory.utils.episode.episode import Episode
from forge.sdk.memory.utils.embeddings import get_ada_embedding
from forge.sdk.config.config import Config
from forge.utils.process_tokens import count_string_tokens


DEF_SCHEMA = {
    "classes": [
        {
            "class": "Episode",
            "description": "Episode",
            "properties": [
                {
                    "name": "meta_episode",
                    "dataType": ["Episode"],
                    "description": "The episode which generalizes this episode",
                },
                {
                    "name": "overview",
                    "dataType": ["text"],
                    "description": "The overview of the episode",
                },
                {
                    "name": "goal",
                    "dataType": ["text"],
                    "description": "Goal of the episode",
                },
                {
                    "name": "capability",
                    "dataType": ["text"],
                    "description": "The capability that was used in the episode",
                },
                {
                    "name": "ability",
                    "dataType": ["text"],
                    "description": "The ability that was performed in the episode",
                },
                {
                    "name": "arguments",
                    "dataType": ["text"],
                    "description": "The arguments that were used in the episode",
                },
                {
                    "name": "observation",
                    "dataType": ["text"],
                    "description": "The observation of the episode",
                },
                {
                    "name": "child_episodes_uuid",
                    "dataType": ["text[]"],
                    "description": "The uuids of the child episodes",
                },
                {
                    "name": "created_at",
                    "dataType": ["text"],
                    "description": "The date the episode was created",
                },
            ],
        },
        {
            "class": "WebData",
            "description": "WebData",
            "properties": [
                {
                    "name": "url",
                    "dataType": ["text"],
                    "description": "The url where the data was found",
                },
                {
                    "name": "query",
                    "dataType": ["text"],
                    "description": "The query used to find the data",
                },
                {
                    "name": "content",
                    "dataType": ["text"],
                    "description": "The content of the data",
                },
            ],
        }
    ]
}


class WeaviateMemory(object):
    def __init__(self):
        weaviate_key = Config().weaviate_key
        if weaviate_key:
            # Run on weaviate cloud service
            auth = weaviate.auth.AuthApiKey(api_key=weaviate_key)
            self.client = weaviate.Client(
                url=Config().weaviate_url,
                auth_client_secret=auth,
                additional_headers={
                    "X-OpenAI-Api-Key": Config().openai_api_key,
                },
            )
        else:
            # Run locally"
            self.client = weaviate.Client(
                url=f"{Config().local_weaviate_url}:{Config().weaviate_port}"
            )
        # self.client.schema.delete_all()
        self._create_schema()

    def _create_schema(self):
        # Check if classes in the schema already exist in Weaviate
        for class_definition in DEF_SCHEMA["classes"]:
            class_name = class_definition["class"]
            print("Creating class: ", class_name)
            try:
                if not self.client.schema.contains(class_definition):
                    # Class doesn't exist, so we attempt to create it
                    self.client.schema.create_class(class_definition)
            except Exception as err:
                print(f"Unexpected error {err=}, {type(err)=}")

    def _get_overview_filter(self, overview):
        filter_obj = {
            "path": ["overview", "overview", "name"],
            "operator": "Equal",
            "valueText": overview,
        }
        return filter_obj

    def _get_relevant(
        self,
        vector,
        class_name,
        fields,
        where_filter=None,
        num_relevant=2,
    ):
        try:
            query = (
                self.client.query.get(class_name, fields)
                .with_near_vector(vector)
                .with_limit(num_relevant)
                .with_additional(["certainty", "id"])
            )
            if where_filter:
                query.with_where(where_filter)
            results = query.do()

            if len(results["data"]["Get"][class_name]) > 0:
                return results["data"]["Get"][class_name]
            else:
                return None

        except Exception as err:
            print(f"Unexpected error {err=}, {type(err)=}")
            return None

    def retrieve_episode(self, episode_uuid):
        try:
            query = self.client.query.get(
                "Episode", ["overview", "content", "child_episodes_uuid", "created_at"]
            ).with_additional(["id"])

            query.with_where(self._get_id_filter(id=episode_uuid))
            results = query.do()

            if len(results["data"]["Get"]["Episode"]) > 0:
                return results["data"]["Get"]["Episode"][0]
            else:
                return None
        except Exception as err:
            print(f"Unexpected error {err=}, {type(err)=}")
            return None

    def get_episode(self, episode_uuid: str) -> Optional[Episode]:
        stored_episode = self.retrieve_episode(
            episode_uuid=episode_uuid
        )
        if stored_episode:
            child_episodes = []
            for child_episode_uuid in stored_episode["child_episodes_uuid"]:
                child_episodes.append(
                    self.get_episode(
                        episode_uuid=child_episode_uuid
                    )
                )
            episode = Episode(
                overview=stored_episode["overview"], content=stored_episode["content"]
            )
            episode._creation_time = stored_episode["created_at"]
            episode.add_child_episodes(episodes=child_episodes)
            episode.link_to_uuid(uuid=episode_uuid)
            return episode
        return None

    def search_episode(
        self, query, num_relevant=1, certainty=0.9
    ) -> Optional[Episode]:
        vector = get_ada_embedding(query)
        # Get the most similar overview
        most_similar_contents = self._get_relevant(
            vector=({"vector": vector, "certainty": certainty}),
            class_name="Episode",
            fields=["overview", "goal", "capability", "ability", "arguments", "observation", "child_episodes_uuid", "created_at"],
            num_relevant=num_relevant,
        )
        if most_similar_contents:
            stored_episode = most_similar_contents[0]
            return self.get_episode(
                episode_uuid=stored_episode["_additional"]["id"]
            )
        return None

    def create_episode(
        self,
        overview: str,
        goal: str,
        capability: str,
        ability: str,
        arguments: str,
        observation: str,
        created_at,
        child_episodes_uuid,
    ):
        value = overview  # Using as vector only the overview for now.
        vector = get_ada_embedding(value)
        episode_uuid = self.client.data_object.create(
            data_object={
                "overview": overview,
                "goal": goal,
                "capability": capability,
                "ability": ability,
                "arguments": arguments,
                "observation": observation,
                "created_at": created_at,
                "child_episodes_uuid": child_episodes_uuid,
            },
            class_name="Episode",
            vector=vector,
        )
        return episode_uuid

    def reset_web_data(self):
        self.client.schema.delete_class("WebData")
        self._create_schema()

    def store_web_data(
        self,
        url: str,
        query: str,
        content: str,
    ):
        tokens = count_string_tokens(content)
        if tokens > 2000:
            print(f"Bug!, got {tokens} tokens... edge case that I couldn't get on time.")
            return None

        # TODO: If the object doesn't exist, proceed with creating a new one
        content_vector = get_ada_embedding(content)
        web_data_uuid = self.client.data_object.create(
            data_object={
                "url": url,
                "query": query,
                "content": content,
            },
            class_name="WebData",
            vector=content_vector,
        )
        return web_data_uuid

    def search_web_data(
        self, query, num_relevant=1, certainty=0.7
    ):
        query_vector = get_ada_embedding(query)
        # Get the most similar content
        most_similar_contents = self._get_relevant(
            vector=({"vector": query_vector}),  # TODO: add "certainty": certainty
            class_name="WebData",
            fields=["url", "query", "content"],
            num_relevant=num_relevant,
        )
        return most_similar_contents

    def store_episode(
        self,
        overview: str,
        goal: str,
        capability: str,
        ability: str,
        arguments: str,
        observation: str,
        created_at: str,
        child_episodes_uuid: List[str] = [],
    ):
        episode_uuid = self.create_episode(
            overview=overview,
            goal=goal,
            capability=capability,
            ability=ability,
            arguments=arguments,
            observation=observation,
            created_at=created_at,
            child_episodes_uuid=child_episodes_uuid,
        )
        for child_episode_uuid in child_episodes_uuid:
            # Add parent as cross-reference for each child
            self.update_cross_reference(
                uuid=child_episode_uuid,
                reference_object_uuid=episode_uuid,
                field_name="meta_episode",
                class_name="Episode",
                cross_reference_name="Episode",
                override=True,
            )
        return episode_uuid

    def update_cross_reference(
        self,
        uuid: str,
        reference_object_uuid: str,
        field_name: str,
        class_name: str,
        cross_reference_name: str,
        override: bool = True,
    ):
        pass
        # Create a new instance of the cross reference class
        #if override:
            #self.client.data_object.reference.update(
            #    from_uuid=uuid,
            #    from_property_name=field_name,
            #    to_uuids=[reference_object_uuid],
            #    from_class_name=class_name,
            #    to_class_names=cross_reference_name,
            #)
        # else:
            #self.client.data_object.reference.add(
            #    from_uuid=uuid,
            #    from_property_name=field_name,
            #    to_uuid=reference_object_uuid,
            #    from_class_name=class_name,
            #    to_class_name=cross_reference_name,
            # )

    def recursive_search(self, overview, query, certainty=0.9, depth=1):
        """This method uses remember to first search for the parent overview and then do a recursive search"""

        # First search for most similar episode
        final_query = f"{overview}: {query}"
        vector = get_ada_embedding(final_query)
        # Get the most similar overview
        result = self._get_relevant(
            vector=({"vector": vector, "certainty": certainty}),
            class_name="Episode",
            fields=["overview, content"],
            num_relevant=1,
        )
        episodes = []
        if result:
            episodes.append(result[0]["content"])
            episodes.extend(
                self.remember(final_query, result[0]["overview"], certainty, depth)
            )
        return episodes

    def remember(self, query, parent_overview, certainty=0.9, depth=1):
        """
        Traverse a tree of episodes, searching for the most similar content to the given query at each level.

        This function starts from the given parent_overview and traverses the tree to a specified depth,
        looking for episodes with content similar to the query. The similarity is determined by embeddings
        and a specified certainty threshold.

        Args:
            query (str): The query to find similar content for.
            parent_overview (str): The overview to start the traversal from.
            certainty (float, optional): The similarity threshold for content, default is 0.9.
            depth (int, optional): The maximum depth of the traversal, default is 1.

        Returns:
            list: A list of relevant content found during the traversal.
        """

        episodes = []
        vector = get_ada_embedding(query)
        while depth > 0:
            result = self._get_relevant(
                vector=({"vector": vector, "certainty": certainty}),
                class_name="Episode",
                fields=["overview, content"],
                num_relevant=1,
                where_filter=self._get_meta_episode_overview_filter(parent_overview),
            )
            if result:
                parent_overview = result[0]["overview"]
                relevant_content = result[0]["content"]
                episodes.append(relevant_content)
                depth -= 1
            else:
                break
        return episodes

    def _get_meta_episode_overview_filter(self, overview):
        filter_obj = {
            "path": ["meta_episode", "Episode", "overview"],
            "operator": "Equal",
            "valueText": overview,
        }
        return filter_obj

    def _get_id_filter(self, id):
        id_filter_object = {
            "path": ["id"],
            "operator": "Equal",
            "valueString": id,
        }
        return id_filter_object
