import tiktoken
import subprocess
import spacy
from spacy.util import get_package_path
from typing import List, Optional

from forge.sdk import ForgeLogger

LOG = ForgeLogger(__name__)


def count_string_tokens(string: str, model_name: str = "gpt-3.5-turbo") -> int:
    """
    Returns the number of tokens in a text string.

    Args:
    string (str): The text string.
    model_name (str): The name of the encoding to use. (e.g., "gpt-3.5-turbo")

    Returns:
    int: The number of tokens in the text string.
    """
    encoding = tiktoken.encoding_for_model(model_name)
    return len(encoding.encode(string))


def preprocess_text(
    raw_prompt: str,
    text: str,
    chunk_max_tokens: int,
    prefix: Optional[str] = None,
    model: str = "gpt-3.5-turbo",
) -> List[str]:
    """Preprocess text"""

    model_name = "en_core_web_sm"
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
    LOG.info(f"Total tokens: {sentences_length}")
    if sentences_length > 20000:  # Avoid huge cost!!
        chunk = "This webpage is too long, please search at another source."
        # TODO: Remove me when model is free.
        return [chunk]
    prompt_tokens = count_string_tokens(string=raw_prompt, model_name=model)
    if prefix:
        prompt_tokens = prompt_tokens + count_string_tokens(
            string=prefix, model_name=model
        )
    chunk_max_tokens = chunk_max_tokens - prompt_tokens

    chunks = [sentences.pop(0)]
    for sentence in sentences:
        # Accumulate on current chunk until max tokens is reached.
        future_chunk = chunks[-1] + " " + sentence
        if count_string_tokens(string=future_chunk) >= chunk_max_tokens:
            overlap_text = get_tokens(
                text=chunks[-1], max_tokens=100  # TODO: Move to cfg.
            )
            chunks.append(overlap_text + " " + sentence)
        else:
            chunks[-1] = future_chunk
    LOG.info(f"Number of chunks: {len(chunks)}")
    return chunks


def get_tokens(text: str, max_tokens: int):
    """Get words from text"""
    words = text.split()
    tokens = ""
    for word in words:
        future_tokens = tokens + word + " "
        if count_string_tokens(string=future_tokens) >= max_tokens:
            return tokens.strip()
        tokens = future_tokens
    return tokens.strip()
