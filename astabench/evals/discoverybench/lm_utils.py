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
    "temperature": 0,
    "max_tokens": 250,
    "top_p": 1.0,
    "frequency_penalty": 0,
    "presence_penalty": 0,
}

ANTHROPIC_GEN_HYP = {
    "temperature": 0,
    "max_tokens": 250,
    # Anthropic rejects requests with both `temperature` and `top_p` set
    # ("`temperature` and `top_p` cannot both be specified for this model").
    # Keep `temperature=0` (deterministic judging) and drop `top_p`.
}

any_json_schema = ResponseSchema(
    name="json",
    json_schema=json_schema({"type": "object", "additionalProperties": True}),
)

DEFAULT_EVAL_OPENAI_MODEL = "claude-haiku-4-5"


def _resolve_provider(model: str) -> tuple[str, dict]:
    if "/" in model:
        prefix = model.split("/", 1)[0]
        hyp = ANTHROPIC_GEN_HYP if prefix == "anthropic" else OPENAI_GEN_HYP
        return model, hyp
    if "claude" in model:
        return f"anthropic/{model}", ANTHROPIC_GEN_HYP
    return f"openai/{model}", OPENAI_GEN_HYP


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
    qualified, hyp = _resolve_provider(model_name)
    # Anthropic rejects the structured-output schema we use for OpenAI
    # ("output_format.schema: Invalid schema: Schema type is missing"); skip
    # response_schema for anthropic and rely on the freeform-JSON parser in
    # `get_response()` (which already tolerates fenced/preamble output from
    # reasoning models).
    use_schema = json_response and not qualified.startswith("anthropic/")
    model = get_model(
        qualified,
        config=GenerateConfig(
            response_schema=any_json_schema if use_schema else None, **hyp
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
    qualified, hyp = _resolve_provider(model)
    model = get_model(qualified, config=GenerateConfig(**hyp))
    n_try = 0
    while n_try < max_retry:
        response = await model.generate(oaidicts_to_chatmessages(create_prompt(prompt)))
        # Reasoning/thinking models (Claude Haiku 4.5, gpt-5-mini, ...) return
        # message.content as a list of typed parts (ContentReasoning,
        # ContentText, ...) rather than a flat string.  Concatenate the
        # text-bearing parts before calling .strip().
        raw_content = response.choices[0].message.content
        if isinstance(raw_content, list):
            raw_content = "".join(
                part.text for part in raw_content
                if hasattr(part, "text") and part.text
            )
        output = raw_content.strip().strip("`json")
        try:
            response_json = json.loads(output)
            return response_json
        except ValueError:
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
