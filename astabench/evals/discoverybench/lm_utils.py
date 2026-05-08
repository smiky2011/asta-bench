import json

from inspect_ai.model import (
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageUser,
    GenerateConfig,
    ResponseSchema,
    get_model,
)
from inspect_ai.util import json_schema

OPENAI_GEN_HYP = {
    "temperature": 1,
    "max_tokens": 250,
    "top_p": 1.0,
    "frequency_penalty": 0,
    "presence_penalty": 0,
}

any_json_schema = ResponseSchema(
    name="json",
    json_schema=json_schema({"type": "object", "additionalProperties": True}),
)

DEFAULT_EVAL_OPENAI_MODEL = "gpt-5-mini"


def oaidicts_to_chatmessages(
    messages: list[dict[str, str]],
) -> list[ChatMessageUser | ChatMessageSystem | ChatMessageAssistant]:
    """Convert OpenAI API message dicts to ChatMessage objects."""
    chat_messages = []
    for message in messages:
        role = message["role"]
        content = message["content"]
        if role == "user":
            chat_messages.append(ChatMessageUser(content=content))
        elif role == "system":
            chat_messages.append(ChatMessageSystem(content=content))
        elif role == "assistant":
            chat_messages.append(ChatMessageAssistant(content=content))
        else:
            raise ValueError(f"Unknown role: {role}")
    return chat_messages


async def run_chatgpt_query_multi_turn(
    messages,
    model_name=DEFAULT_EVAL_OPENAI_MODEL,
    json_response=False,
):
    model = get_model(
        f"openai/{model_name}",
        config=GenerateConfig(
            response_schema=any_json_schema if json_response else None, **OPENAI_GEN_HYP
        ),
    )
    return await model.generate(oaidicts_to_chatmessages(messages))


def create_prompt(usr_msg):
    return [
        {
            "role": "system",
            "content": "You are a helpful assistant who is not talkative. You only respond with the exact answer to a query without additional conversation.",
        },
        {"role": "user", "content": usr_msg},
    ]


async def get_response(
    prompt: str,
    max_retry: int = 5,
    model: str = DEFAULT_EVAL_OPENAI_MODEL,
    verbose=False,
):
    model = get_model(f"openai/{model}", config=GenerateConfig(**OPENAI_GEN_HYP))
    n_try = 0
    while n_try < max_retry:
        response = await model.generate(oaidicts_to_chatmessages(create_prompt(prompt)))
        # gpt-5-mini (reasoning model) returns content as a list of typed
        # parts (ContentText, ContentReasoning, ...) rather than a flat str
        # like gpt-4o.  Concatenate text parts before parsing.
        raw_content = response.choices[0].message.content
        if isinstance(raw_content, list):
            raw_content = "".join(
                part.text for part in raw_content if hasattr(part, "text") and part.text
            )
        # gpt-5-mini may emit reasoning preamble + a fenced ```json``` block
        # or a bare JSON object.  Try plain parse first; fall back to
        # carving out the first JSON object found.
        output = raw_content.strip()
        try:
            return json.loads(output)
        except ValueError:
            pass
        import re as _re
        fenced = _re.search(r"```(?:json)?\s*(\{.*?\})\s*```", output, flags=_re.DOTALL)
        candidate = fenced.group(1) if fenced else None
        if candidate is None:
            try:
                start = output.index("{")
                end = output.rindex("}")
                candidate = output[start : end + 1]
            except ValueError:
                candidate = None
        if candidate is not None:
            try:
                return json.loads(candidate)
            except ValueError:
                pass
        if verbose:
            print(f"Bad JSON output:\n\n{output}")
        n_try += 1
        if n_try < max_retry:
            if verbose:
                print("Retrying...")
        else:
            if verbose:
                print("Retry limit reached")
    return None
