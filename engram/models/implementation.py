"""Implementation configuration model."""

from dataclasses import dataclass, field


@dataclass
class ConfigManagement:
    """How config is managed for this implementation."""

    mode: str = 'local'  # 'local' or 'pull-push'
    workflow_id: str = ''


@dataclass
class ImplementationConfig:
    """Implementation definition from implementation.yaml."""

    workflow: str
    platform: str
    runner: str
    runner_config: dict[str, str] = field(default_factory=dict)
    config_management: ConfigManagement = field(default_factory=ConfigManagement)
