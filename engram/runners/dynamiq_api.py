"""Shared Dynamiq management API client."""

from __future__ import annotations

import os
from typing import Any

import httpx

DYNAMIQ_API_BASE = 'https://api.getdynamiq.ai/v1'


def management_api(
    jwt_env: str,
    path: str,
    params: dict | None = None,
    method: str = 'GET',
    json_body: dict | None = None,
) -> dict[str, Any]:
    """
    Call the Dynamiq management API.

    Args:
        jwt_env: Name of the env var holding the JWT token.
        path: API path (e.g. '/apps/{id}').
        params: Query parameters.
        method: HTTP method.
        json_body: JSON request body for POST/PUT.

    """
    jwt = os.environ.get(jwt_env, '') if jwt_env else os.environ.get('DYNAMIQ_JWT_TOKEN', '')
    headers = {
        'accept': 'application/json',
        'authorization': f'Bearer {jwt}',
    }
    resp = httpx.request(
        method, f'{DYNAMIQ_API_BASE}{path}', params=params, json=json_body, headers=headers, timeout=30
    )
    resp.raise_for_status()
    return resp.json()
