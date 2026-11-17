"""The one place that creates the boto3 `bedrock-runtime` client, plus retries."""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from typing import Literal

from semcache.config import Settings, get_settings

AuthMode = Literal["api_key", "iam"]
BEARER_ENV_VAR = "AWS_BEARER_TOKEN_BEDROCK"

RETRYABLE_CODES = {
    "ThrottlingException",
    "TooManyRequestsException",
    "ServiceUnavailableException",
    "ModelNotReadyException",
    "InternalServerException",
}

_clients: dict[tuple[str, AuthMode], object] = {}


class BedrockAuthError(RuntimeError):
    """Credentials are missing or ambiguous. Messages name env vars, never their values."""


def auth_mode(settings: Settings) -> AuthMode:
    token = settings.aws_bearer_token_bedrock.get_secret_value().strip()
    key_id = settings.aws_access_key_id.get_secret_value().strip()
    secret = settings.aws_secret_access_key.get_secret_value().strip()
    if token and (key_id or secret):
        # Refuse instead of guessing: an unexpected IAM key may have far wider permissions.
        raise BedrockAuthError(
            f"Both {BEARER_ENV_VAR} and AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY are set. "
            "Keep only one auth option in .env."
        )
    if token:
        return "api_key"
    if key_id and secret:
        return "iam"
    raise BedrockAuthError(
        f"No Bedrock credentials. Set {BEARER_ENV_VAR} (a Bedrock API key) or "
        "AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY in .env."
    )


def get_runtime_client(settings: Settings | None = None):
    """A cached `bedrock-runtime` client for the configured region."""
    import boto3
    from botocore.config import Config

    settings = settings or get_settings()
    mode = auth_mode(settings)
    key = (settings.aws_region, mode)
    if key in _clients:
        return _clients[key]
    config = Config(
        retries={"max_attempts": 6, "mode": "standard"}, connect_timeout=10, read_timeout=120
    )
    if mode == "api_key":
        # boto3 reads the API key only from the process environment.
        os.environ[BEARER_ENV_VAR] = settings.aws_bearer_token_bedrock.get_secret_value().strip()
        client = boto3.client("bedrock-runtime", region_name=settings.aws_region, config=config)
    else:
        # An *empty* bearer variable counts as "a token exists" to botocore; remove it.
        if not os.environ.get(BEARER_ENV_VAR, "").strip():
            os.environ.pop(BEARER_ENV_VAR, None)
        client = boto3.client(
            "bedrock-runtime",
            region_name=settings.aws_region,
            aws_access_key_id=settings.aws_access_key_id.get_secret_value().strip(),
            aws_secret_access_key=settings.aws_secret_access_key.get_secret_value().strip(),
            config=config,
        )
    _clients[key] = client
    return client


def with_retries[T](
    call: Callable[[], T], *, attempts: int = 4, sleep: Callable[[float], None] = time.sleep
) -> T:
    """Run `call`, retrying retryable AWS errors with growing pauses (1 s, 2 s, 4 s)."""
    from botocore.exceptions import ClientError

    for attempt in range(1, attempts + 1):
        try:
            return call()
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            if code not in RETRYABLE_CODES or attempt == attempts:
                raise
            sleep(2.0 ** (attempt - 1))
    raise AssertionError("unreachable")
