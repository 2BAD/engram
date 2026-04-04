"""
Config sync for hosted platforms: pull, push, diff, deploy.

Operates on ImplementationConfig directly. Reads workflow_id and jwt_env
from config_management, app_id from runner_config.
"""

from __future__ import annotations

import copy
import json
from datetime import UTC, datetime
from difflib import unified_diff
from pathlib import Path
from typing import TYPE_CHECKING, Any

from engram.runners.dynamiq_api import management_api

if TYPE_CHECKING:
    from engram.models.implementation import ImplementationConfig


# -- Public API --


def pull_config(impl_dir: Path, impl_config: ImplementationConfig) -> dict[str, Any]:
    """Pull workflow config from Dynamiq, write manifest and prompt files."""
    jwt_env = impl_config.config_management.jwt_env
    workflow_id = impl_config.config_management.workflow_id
    app_id = impl_config.runner_config.get('app_id', '')

    wf = _fetch_workflow(jwt_env, workflow_id)
    wf_name = wf.get('name', workflow_id[:12])
    nodes = wf.get('flow', {}).get('nodes', [])

    extracted = _extract_editable_nodes(nodes)
    prompts_dir = impl_dir / 'prompts'
    prompts_dir.mkdir(parents=True, exist_ok=True)

    for node_info in extracted:
        for filename, text in zip(node_info['prompt_files'], node_info['_texts'], strict=True):
            (prompts_dir / filename).write_text(text)

    manifest_nodes = [{k: v for k, v in n.items() if k != '_texts'} for n in extracted]

    manifest = {
        'workflow_id': workflow_id,
        'workflow_name': wf_name,
        'app_id': app_id or None,
        'pulled_at': datetime.now(UTC).isoformat(),
        'nodes': manifest_nodes,
    }

    (impl_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def diff_config(impl_dir: Path, impl_config: ImplementationConfig) -> list[str]:
    """Compare local config vs remote. Returns list of diff description lines."""
    jwt_env = impl_config.config_management.jwt_env
    manifest = _load_manifest(impl_dir)
    workflow_id = manifest['workflow_id']

    remote_nodes = _fetch_remote_node_map(jwt_env, workflow_id)
    lines: list[str] = []

    for node_info in manifest['nodes']:
        nid = node_info['id']
        local_texts = _load_local_texts(impl_dir, node_info)
        local_model = node_info['model']

        if nid not in remote_nodes:
            lines.append(f'{nid}: not found in remote (may have been removed)')
            continue

        remote = remote_nodes[nid]

        if local_model != remote['model']:
            lines.append(f'{nid} model: {remote["model"]} -> {local_model}')

        for i, filename in enumerate(node_info['prompt_files']):
            local_text = local_texts[i] if i < len(local_texts) else ''
            remote_text = remote['texts'][i] if i < len(remote['texts']) else ''
            if local_text != remote_text:
                diff = list(
                    unified_diff(
                        remote_text.splitlines(),
                        local_text.splitlines(),
                        fromfile=f'remote/{filename}',
                        tofile=f'local/{filename}',
                        lineterm='',
                    )
                )
                lines.extend(diff)

    local_ids = {n['id'] for n in manifest['nodes']}
    for nid in remote_nodes:
        if nid not in local_ids:
            lines.append(f'{nid}: exists in remote but not tracked locally')

    return lines


def push_config(impl_dir: Path, impl_config: ImplementationConfig, dry_run: bool = False) -> list[str]:
    """Push local config changes to Dynamiq. Returns list of change descriptions."""
    jwt_env = impl_config.config_management.jwt_env
    manifest = _load_manifest(impl_dir)
    workflow_id = manifest['workflow_id']

    wf = _fetch_workflow(jwt_env, workflow_id)
    flow = copy.deepcopy(wf.get('flow', {}))
    nodes = flow.get('nodes', [])

    changes: list[str] = []

    for local_node in manifest['nodes']:
        nid = local_node['id']
        local_texts = _load_local_texts(impl_dir, local_node)

        flow_node = _find_flow_node(nodes, nid)
        if not flow_node:
            changes.append(f'{nid}: not found in remote, skipping')
            continue

        node_type = local_node['type']
        is_agent = 'Agent' in node_type or 'ReAct' in node_type

        old_model = _get_node_model(flow_node, is_agent)
        model_changed = old_model != local_node['model']
        old_texts = _get_node_texts(flow_node, is_agent)
        prompt_changed = old_texts != local_texts

        if model_changed or prompt_changed:
            desc = []
            if model_changed:
                desc.append(f'model: {old_model} -> {local_node["model"]}')
            if prompt_changed:
                desc.append('prompt changed')
            changes.append(f'{nid}: {", ".join(desc)}')

            if not dry_run:
                _patch_node(flow_node, local_node, local_texts, is_agent)

    if changes and not dry_run:
        save_payload = {**wf, 'flow': flow}
        for key in ('id', 'created_at', 'updated_at'):
            save_payload.pop(key, None)
        management_api(jwt_env, f'/workflows/{workflow_id}/save', method='POST', json_body=save_payload)

    return changes


def deploy_config(impl_dir: Path, impl_config: ImplementationConfig, dry_run: bool = False) -> list[str]:
    """Push changes then deploy to the app. Returns change descriptions."""
    changes = push_config(impl_dir, impl_config, dry_run=dry_run)

    if dry_run:
        return changes

    jwt_env = impl_config.config_management.jwt_env
    manifest = _load_manifest(impl_dir)
    workflow_id = manifest['workflow_id']
    app_id = impl_config.runner_config.get('app_id', '') or manifest.get('app_id', '')

    if not app_id:
        changes.append('deploy skipped: no app_id configured')
        return changes

    versions = management_api(
        jwt_env, '/workflow-versions', params={'workflow_id': workflow_id, 'page': 1, 'page_size': 1}
    )
    version_list = versions.get('data', [])
    if not version_list:
        changes.append('deploy skipped: no workflow versions found')
        return changes
    version_id = version_list[0]['id']

    deployments = management_api(jwt_env, f'/apps/{app_id}/deployments')
    deploys = deployments.get('data', [])
    runtime_id = deploys[0].get('runtime_id', '') if deploys else ''

    if not runtime_id:
        runtimes = management_api(jwt_env, '/runtimes')
        runtime_list = runtimes.get('data', [])
        runtime_id = runtime_list[0]['id'] if runtime_list else ''

    if not runtime_id:
        changes.append('deploy skipped: no runtime available')
        return changes

    management_api(
        jwt_env,
        f'/apps/{app_id}/deploy',
        method='POST',
        json_body={
            'workflow_id': workflow_id,
            'workflow_version_id': version_id,
            'runtime_id': runtime_id,
            'deployment_config': {'deployment_type': 'serverless'},
        },
    )

    changes.append(f'deployed version {version_id[:16]}')
    return changes


# -- Internal helpers --


def _fetch_workflow(jwt_env: str, workflow_id: str) -> dict[str, Any]:
    resp = management_api(jwt_env, f'/workflows/{workflow_id}')
    return resp.get('data', resp)


def _load_manifest(impl_dir: Path) -> dict[str, Any]:
    return json.loads((impl_dir / 'manifest.json').read_text())


def _load_local_texts(impl_dir: Path, node_info: dict) -> list[str]:
    prompts_dir = impl_dir / 'prompts'
    return [(prompts_dir / f).read_text() if (prompts_dir / f).exists() else '' for f in node_info['prompt_files']]


def _fetch_remote_node_map(jwt_env: str, workflow_id: str) -> dict[str, dict]:
    wf = _fetch_workflow(jwt_env, workflow_id)
    nodes = wf.get('flow', {}).get('nodes', [])
    extracted = _extract_editable_nodes(nodes)
    return {n['id']: {'model': n['model'], 'texts': n['_texts']} for n in extracted}


def _extract_editable_nodes(nodes: list[dict]) -> list[dict[str, Any]]:
    """Walk flow nodes and extract editable LLM/Agent node configs."""
    results = []
    for node in nodes:
        node_type = node.get('type', '')
        if 'Map' in node_type:
            inner = node.get('node', {})
            if isinstance(inner, dict):
                extracted = _classify_editable_node(inner, map_id=node.get('id', ''))
                if extracted:
                    results.append(extracted)
            continue
        extracted = _classify_editable_node(node)
        if extracted:
            results.append(extracted)
    return results


def _classify_editable_node(node: dict, map_id: str | None = None) -> dict[str, Any] | None:
    """Classify a node and extract its editable config with prompt file mappings."""
    node_type = node.get('type', '')
    effective_id = map_id or node.get('id', '')

    if any(t in node_type for t in ('OpenAI', 'Anthropic', 'LLM')):
        messages = node.get('prompt', {}).get('messages', [])
        prompt_files = []
        message_roles = []
        content_formats = []
        texts = []

        for msg in messages:
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')

            if isinstance(content, list):
                text = '\n'.join(
                    block.get('text', '') for block in content if isinstance(block, dict) and block.get('text')
                )
                content_formats.append('content_blocks')
            else:
                text = str(content)
                content_formats.append('string')

            prompt_files.append(f'{effective_id}.{role}.md')
            message_roles.append(role)
            texts.append(text)

        return {
            'id': effective_id,
            'name': node.get('name', effective_id),
            'type': node_type,
            'model': node.get('model', ''),
            'is_map': map_id is not None,
            'prompt_files': prompt_files,
            'message_roles': message_roles,
            'content_formats': content_formats,
            '_texts': texts,
        }

    if 'Agent' in node_type or 'ReAct' in node_type:
        llm = node.get('llm', {})
        model = llm.get('model', '') if isinstance(llm, dict) else node.get('model', '')
        role_text = node.get('role', '') or ''

        return {
            'id': effective_id,
            'name': node.get('name', effective_id),
            'type': node_type,
            'model': model,
            'is_map': map_id is not None,
            'prompt_files': [f'{effective_id}.role.md'],
            'message_roles': ['role'],
            'content_formats': ['string'],
            '_texts': [role_text],
        }

    return None


def _find_flow_node(nodes: list[dict], target_id: str) -> dict | None:
    """Find a node in the flow by ID, unwrapping Map nodes."""
    for node in nodes:
        node_type = node.get('type', '')
        if 'Map' in node_type and node.get('id', '') == target_id:
            inner = node.get('node', {})
            return inner if isinstance(inner, dict) else None
        if node.get('id', '') == target_id:
            return node
    return None


def _get_node_model(flow_node: dict, is_agent: bool) -> str:
    if is_agent:
        llm = flow_node.get('llm', {})
        return llm.get('model', '') if isinstance(llm, dict) else flow_node.get('model', '')
    return flow_node.get('model', '')


def _get_node_texts(flow_node: dict, is_agent: bool) -> list[str]:
    if is_agent:
        return [flow_node.get('role', '')]
    return [_extract_message_text(m) for m in flow_node.get('prompt', {}).get('messages', [])]


def _extract_message_text(msg: dict) -> str:
    content = msg.get('content', '')
    if isinstance(content, list):
        return '\n'.join(block.get('text', '') for block in content if isinstance(block, dict) and block.get('text'))
    return str(content)


def _patch_node(flow_node: dict, local_info: dict, local_texts: list[str], is_agent: bool) -> None:
    """Mutate a flow node with local model and prompt changes."""
    if is_agent:
        llm = flow_node.get('llm', {})
        if isinstance(llm, dict):
            llm['model'] = local_info['model']
        else:
            flow_node['model'] = local_info['model']
        if local_texts:
            flow_node['role'] = local_texts[0]
    else:
        flow_node['model'] = local_info['model']
        messages = flow_node.get('prompt', {}).get('messages', [])
        for i, text in enumerate(local_texts):
            if i >= len(messages):
                break
            fmt = local_info['content_formats'][i] if i < len(local_info['content_formats']) else 'string'
            if fmt == 'content_blocks':
                messages[i]['content'] = [{'type': 'text', 'text': text}]
            else:
                messages[i]['content'] = text
