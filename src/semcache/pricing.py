"""On-demand Bedrock prices and per-call cost (SPEC section 2).

All prices are USD for ap-south-1 (Mumbai), taken from the public AWS Price List and the
Bedrock pricing page. Prices change: update this table and PRICES_CHECKED_ON together.
"""

from dataclasses import dataclass
from datetime import date

PRICES_CHECKED_ON = date(2026, 11, 16)
PRICE_REGION = "ap-south-1"


@dataclass(frozen=True)
class ModelPrice:
    input_per_1m: float = 0.0  # USD per 1M input tokens
    output_per_1m: float = 0.0  # USD per 1M output tokens


PRICES: dict[str, ModelPrice] = {
    "amazon.titan-embed-text-v2:0": ModelPrice(input_per_1m=0.024),
    # Global cross-region inference profile, billed at the global rate.
    "global.amazon.nova-2-lite-v1:0": ModelPrice(input_per_1m=0.35, output_per_1m=2.95),
    # Upgrade path (needs AWS Marketplace access): a Claude Haiku class model.
    "global.anthropic.claude-haiku-5-5": ModelPrice(input_per_1m=0.10, output_per_1m=0.50),
    # Used by tests and the traffic simulation's mock upstream (priced like Nova 2 Lite).
    "mock-upstream": ModelPrice(input_per_1m=0.35, output_per_1m=2.95),
    "mock-embed": ModelPrice(input_per_1m=0.024),
}


class UnknownModelPriceError(KeyError):
    """A model ID with no price. Raised so cost accounting can never silently show $0."""


def price_for(model_id: str) -> ModelPrice:
    try:
        return PRICES[model_id]
    except KeyError:
        raise UnknownModelPriceError(
            f"No price for model {model_id!r}. Add it to semcache/pricing.py."
        ) from None


def cost_usd(model_id: str, input_tokens: int = 0, output_tokens: int = 0) -> float:
    """Estimated USD cost of one call to `model_id`."""
    if input_tokens < 0 or output_tokens < 0:
        raise ValueError("token counts cannot be negative")
    p = price_for(model_id)
    return input_tokens * p.input_per_1m / 1_000_000 + output_tokens * p.output_per_1m / 1_000_000
