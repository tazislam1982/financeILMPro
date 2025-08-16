from typing import Generator, Union, Dict, Any
from openai import OpenAI, APIConnectionError, RateLimitError, APIStatusError
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from dotenv import load_dotenv
import os
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

client = OpenAI()

# Defaults used if not provided via **kwargs
_DEFAULTS: Dict[str, Any] = {
    "model": "gpt-4o-mini",
    "max_tokens": 1200,
    "temperature": 0.1,
}

def parsed_completion_v1(**kwargs) -> Union[ChatCompletion, Generator[ChatCompletionChunk, None, None]]:
    """
    Upgraded but compatible:
    - Keeps function name & **kwargs signature
    - Supports both streaming and non-streaming
    - Merges sensible defaults if not provided
    - Optional: pass include_usage=True to add stream_options={'include_usage': True}
      (or pass your own stream_options dict)
    """
    # merge defaults without clobbering caller-specified values
    merged = {**_DEFAULTS, **kwargs}

    # infer streaming
    stream = bool(merged.pop("stream", False))

    # optional convenience flag: include token usage in streaming tail chunk
    include_usage = bool(merged.pop("include_usage", False))
    if stream and include_usage:
        so = dict(merged.get("stream_options", {}))
        so["include_usage"] = True
        merged["stream_options"] = so

    try:
        if stream:
            return client.chat.completions.create(stream=True, **merged)
        else:
            return client.chat.completions.create(**merged)
    except APIConnectionError as e:
        print("API Connection Error:", e)
        raise
    except RateLimitError as e:
        print("Rate Limit Error:", e)
        raise
    except APIStatusError as e:
        print("API Status Error:", e)
        raise
