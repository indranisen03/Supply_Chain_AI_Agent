"""
LLM factory — returns ChatAnthropic or ChatOpenAI based on config.
Import get_llm() instead of instantiating models directly.
"""

from config import (
    ANTHROPIC_API_KEY,
    MODEL_JUDGE,
    MODEL_LARGE,
    MODEL_MINI,
    OPENAI_API_KEY,
    _USE_CLAUDE_ONLY,
)


def get_llm(model_key: str = "mini", **kwargs):
    """
    model_key: "mini" | "large" | "judge"
    Returns a ready-to-use chat model.
    """
    model_name = {"mini": MODEL_MINI, "large": MODEL_LARGE, "judge": MODEL_JUDGE}[model_key]

    is_claude = model_name.startswith("claude")

    if is_claude:
        from langchain_anthropic import ChatAnthropic
        defaults = {"temperature": 0, "max_tokens": 1024}
        defaults.update(kwargs)
        return ChatAnthropic(model=model_name, api_key=ANTHROPIC_API_KEY, **defaults)
    else:
        from langchain_openai import ChatOpenAI
        defaults = {"temperature": 0}
        defaults.update(kwargs)
        return ChatOpenAI(model=model_name, api_key=OPENAI_API_KEY, **defaults)


def token_cost(tokens: int, model_key: str = "mini") -> float:
    """Approximate cost in USD for a token count."""
    model_name = {"mini": MODEL_MINI, "large": MODEL_LARGE, "judge": MODEL_JUDGE}[model_key]
    # Claude pricing (per 1M tokens, blended input+output estimate)
    rates = {
        "claude-haiku-4-5-20251001": 0.40,
        "claude-sonnet-4-6": 4.50,
        "claude-sonnet-4-5": 4.50,
        "gpt-4o-mini": 0.60,
        "gpt-4o": 6.00,
    }
    rate = rates.get(model_name, 1.0)
    return (tokens / 1_000_000) * rate
