from src.config import stream_completion_kwargs_with_usage
from src.models import Message
from src.prompt import prompts_on_source
from src.services.openaiservice import parsed_completion_v1
from src.config import completion_kwargs
from openai.types.chat.chat_completion import ChatCompletion


def completion_v1(
    context,
    message_history: list[Message],
    referrer: str
) -> ChatCompletion:
    """
    Generates a completion for a given prompt.

    Args:
        context: The context of the prompt, taken from sources.
        message_history: The message history of the conversation.
        referrer: Which IC property the prompt came from.
        score: The relevance score of the context.

    Returns:
        ChatCompletion: The generated chat completion.
        """
    messages = [
        {"role": "system", "content": prompts_on_source(referrer, context[0], context[2])},
    ]

    # Add the last 4 messages to history
    history = message_history[-4:]
    messages.extend(history)
    

    res = parsed_completion_v1(**completion_kwargs, messages=messages)
    return ChatCompletion(**res.__dict__)
    
def completion_v1_stream(context, message_history: list[Message], referrer: str):
    """
    Generates a streaming completion for a given prompt.

    Args:
        context: The context of the prompt, taken from sources.
        message_history: The message history of the conversation.
        referrer: Which IC property the prompt came from.
        score: The relevance score of the context.
    """
    messages = [
        {"role": "system", "content": prompts_on_source(referrer, context[0], context[2])},
    ]
    
    history = message_history[-4:]
    messages.extend(history)


    req_kwargs = stream_completion_kwargs_with_usage
    res = parsed_completion_v1(**req_kwargs, messages=messages)

    return res
