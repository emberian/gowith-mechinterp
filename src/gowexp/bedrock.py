"""Thin AWS Bedrock client over the Converse API (cross-family, no activations).

The Converse API (`bedrock-runtime.converse`) gives one request/response shape
across Anthropic Claude, Amazon Nova, and Mistral, so the black-box replication
arm can treat all three families uniformly. We only need plain-text turns here.

Request shape (validated against the AWS docs + one live call):

    client.converse(
        modelId=<bedrock model id>,
        messages=[{"role": "user", "content": [{"text": <user>}]}],
        system=[{"text": <system>}],               # omitted when system is empty
        inferenceConfig={"maxTokens": .., "temperature": .., "topP": ..},
    )

Response shape:

    resp["output"]["message"]["content"][0]["text"]   # the assistant text
    resp["stopReason"]                                  # end_turn | max_tokens | ...
    resp["usage"] == {"inputTokens": .., "outputTokens": .., "totalTokens": ..}

Transient failures (ThrottlingException, ModelTimeoutException, and the other
retryable server-side errors) are retried with jittered exponential backoff;
client errors (ValidationException, AccessDeniedException) are raised immediately.
"""
from __future__ import annotations

import random
import time
from functools import lru_cache
from typing import Any

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

# Bedrock error codes that are worth retrying (transient / server-side). Anything
# else (validation, access denied, bad model id) is a hard error we surface at once.
_RETRYABLE_CODES = frozenset(
    {
        "ThrottlingException",
        "ModelTimeoutException",
        "ServiceUnavailableException",
        "InternalServerException",
        "ServiceQuotaExceededException",
    }
)


@lru_cache(maxsize=8)
def _client(region: str):
    """Cached bedrock-runtime client per region.

    We lean on botocore's own adaptive retries as a backstop and add an explicit
    jittered backoff loop in `chat` for clearer control over the long grid run.
    A generous read timeout keeps slow 'thorough' generations from tripping the
    socket before the model finishes.
    """
    cfg = Config(
        region_name=region,
        read_timeout=300,
        connect_timeout=10,
        retries={"max_attempts": 3, "mode": "adaptive"},
    )
    return boto3.client("bedrock-runtime", config=cfg)


def _is_retryable(err: ClientError) -> bool:
    code = err.response.get("Error", {}).get("Code", "")
    return code in _RETRYABLE_CODES


def chat(
    model_id: str,
    system: str,
    user: str,
    max_tokens: int,
    temperature: float,
    region: str,
    *,
    top_p: float | None = None,
    max_retries: int = 6,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
) -> dict[str, Any]:
    """One Converse turn. Returns ``{"text", "in_tokens", "out_tokens", "stop_reason"}``.

    Retries transient Bedrock errors with full-jitter exponential backoff. The
    ``system`` block is only sent when non-empty (some families reject an empty
    system block via Converse).
    """
    client = _client(region)

    inference: dict[str, Any] = {"maxTokens": max_tokens, "temperature": temperature}
    if top_p is not None:
        inference["topP"] = top_p

    kwargs: dict[str, Any] = {
        "modelId": model_id,
        "messages": [{"role": "user", "content": [{"text": user}]}],
        "inferenceConfig": inference,
    }
    if system and system.strip():
        kwargs["system"] = [{"text": system}]

    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = client.converse(**kwargs)
            return _parse(resp)
        except ClientError as e:
            last_err = e
            if not _is_retryable(e) or attempt == max_retries - 1:
                raise
            # full-jitter exponential backoff
            delay = min(max_delay, base_delay * (2 ** attempt))
            time.sleep(random.uniform(0.0, delay))
        except Exception as e:  # noqa: BLE001 — surface non-Client errors after a couple of retries
            last_err = e
            if attempt >= 1:
                raise
            time.sleep(random.uniform(0.0, base_delay))

    # Loop only exits via return/raise; this is just for the type checker.
    raise RuntimeError(f"converse failed for {model_id}") from last_err


def _parse(resp: dict[str, Any]) -> dict[str, Any]:
    """Pull text + token usage out of a Converse response."""
    content = resp["output"]["message"]["content"]
    # Concatenate any text blocks (normal case is a single block).
    text = "".join(block.get("text", "") for block in content)
    usage = resp.get("usage", {})
    return {
        "text": text,
        "in_tokens": int(usage.get("inputTokens", -1)),
        "out_tokens": int(usage.get("outputTokens", -1)),
        "stop_reason": resp.get("stopReason", ""),
    }
