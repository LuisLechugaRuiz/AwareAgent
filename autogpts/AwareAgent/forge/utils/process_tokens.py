import tiktoken
import subprocess
import spacy
from spacy.util import get_package_path
from typing import List, Optional

from forge.utils.logger.console_logger import ForgeLogger

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
    # if sentences_length > 20000:  # Avoid huge cost!!
    #     chunk = "This webpage is too long, please search at another source."
    #    # TODO: Remove me when model is free.
    #    return [chunk]
    if raw_prompt:
        prompt_tokens = count_string_tokens(string=raw_prompt, model_name=model)
        if prefix:
            prompt_tokens = prompt_tokens + count_string_tokens(
                string=prefix, model_name=model
            )
        chunk_max_tokens = chunk_max_tokens - prompt_tokens

    chunks = []
    overlap_tokens = 100
    for sentence in sentences:
        # Start by checking the token count of the current sentence alone.
        sentence_tokens = count_string_tokens(string=sentence, model_name=model)

        if sentence_tokens > chunk_max_tokens:
            # The sentence is too long. We need to split it into smaller parts.
            sentence_parts = split_sentence_into_chunks(sentence, chunk_max_tokens, model)

            # We treat each part almost as a 'normal' sentence in the following steps.
            for part in sentence_parts:
                if not chunks:  # If the list is empty, we add the first part.
                    chunks.append(part)
                else:
                    current_chunk = chunks[-1]
                    future_chunk = current_chunk + " " + part
                    future_chunk_tokens = count_string_tokens(string=future_chunk, model_name=model)

                    if future_chunk_tokens > chunk_max_tokens:
                        overlap_text = get_tokens(text=current_chunk, max_tokens=overlap_tokens) + " "
                        chunks.append(overlap_text + part)
                    else:
                        chunks[-1] = future_chunk
        else:
            # For sentences that don't exceed the limit, we proceed as normal.
            if not chunks:
                chunks.append(sentence)
            else:
                current_chunk = chunks[-1]
                future_chunk = current_chunk + " " + sentence
                future_chunk_tokens = count_string_tokens(string=future_chunk, model_name=model)

                if future_chunk_tokens > chunk_max_tokens:
                    overlap_text = get_tokens(text=current_chunk, max_tokens=overlap_tokens) + " "
                    chunks.append(overlap_text + sentence)
                else:
                    chunks[-1] = future_chunk
    LOG.info(f"Number of chunks: {len(chunks)}")
    return chunks


def split_sentence_into_chunks(sentence, max_tokens, model_name):
    """
    Split a sentence into smaller parts, each with a maximum number of tokens.

    Args:
    sentence (str): The sentence to split.
    max_tokens (int): The maximum number of tokens allowed in each part.
    model_name (str): The model name parameter for the token count function.

    Returns:
    list: A list of sentence parts, each complying with the maximum tokens limit.
    """
    sentence_parts = []
    words = sentence.split()  # Split the sentence into words.
    current_part = ""

    for word in words:
        # Check if adding the next word would exceed the max tokens for the current part.
        if count_string_tokens(string=current_part + " " + word, model_name=model_name) > max_tokens:
            # If so, we add the current_part to our list (if it's not empty) and start a new one.
            if current_part:
                sentence_parts.append(current_part.strip())
            current_part = word  # Start the new part with the current word.
        else:
            # Otherwise, keep adding words to the current part.
            current_part += " " + word

    # Add the last part if it's not empty.
    if current_part.strip():
        sentence_parts.append(current_part.strip())

    return sentence_parts


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


def indent(text, spaces=4):
    indentation = " " * spaces
    return "\n".join(indentation + line for line in text.split("\n"))
