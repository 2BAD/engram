"""Shared Dynamiq management API client."""

from __future__ import annotations

import json
import os
from pathlib import Path
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


def fetch_deployment_timeline(jwt_env: str, app_id: str) -> list[dict[str, Any]]:
    """Fetch all deployments for an app, sorted oldest-first.

    Returns a list of dicts with workflow_version_id, version number,
    and started_at for building time ranges.
    """
    deployments = []
    page = 1
    while True:
        resp = management_api(jwt_env, f'/apps/{app_id}/deployments', {
            'page': page,
            'page_size': 100,
        })
        deployments.extend(resp.get('data', []))
        total = resp.get('pagination', {}).get('total_count', 0)
        if not resp.get('data') or page * 100 >= total:
            break
        page += 1

    timeline = []
    for d in deployments:
        wv = d.get('workflow_version', {})
        timeline.append({
            'workflow_version_id': wv.get('id', ''),
            'version': wv.get('version'),
            'deployed_at': d.get('started_at', ''),
        })

    timeline.sort(key=lambda d: d['deployed_at'])
    return timeline


def match_trace_version(trace_started_at: str, timeline: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Match a trace timestamp to a deployment version using the timeline.

    Each deployment owns the window from its started_at until the next
    deployment's started_at. The last deployment owns everything after it.
    """
    if not timeline:
        return None

    matched = None
    for deployment in timeline:
        if trace_started_at >= deployment['deployed_at']:
            matched = deployment
    return matched


def get_trace(jwt_env: str, trace_id: str, cache_dir: Path | None = None) -> dict[str, Any]:
    """Fetch full trace detail, serving from local cache when available.

    Caches to {cache_dir}/traces/{trace_id}.json so repeated lookups
    (cost analysis, debugging) don't hit the API again.
    """
    if cache_dir is None:
        cache_dir = Path('data') / 'cache'
    trace_dir = cache_dir / 'traces'
    cached = trace_dir / f'{trace_id}.json'
    if cached.exists():
        return json.loads(cached.read_text())

    result = management_api(jwt_env, f'/tracing/traces/{trace_id}')
    trace_dir.mkdir(parents=True, exist_ok=True)
    cached.write_text(json.dumps(result))
    return result
