"""Project structure discovery."""

from pathlib import Path


def find_project_root(start: Path | None = None) -> Path | None:
    """Walk upward from start (default cwd) looking for engram.yaml."""
    current = start or Path.cwd()
    for parent in [current, *current.parents]:
        if (parent / 'engram.yaml').exists():
            return parent
    return None


def discover_workflows(root: Path) -> list[str]:
    """Find all workflow directories containing workflow.yaml."""
    workflows_dir = root / 'workflows'
    if not workflows_dir.exists():
        return []
    return sorted(d.name for d in workflows_dir.iterdir() if d.is_dir() and (d / 'workflow.yaml').exists())


def discover_implementations(root: Path) -> list[str]:
    """Find all implementation directories containing implementation.yaml."""
    impl_dir = root / 'implementations'
    if not impl_dir.exists():
        return []
    return sorted(d.name for d in impl_dir.iterdir() if d.is_dir() and (d / 'implementation.yaml').exists())


def discover_datasets(root: Path) -> list[str]:
    """Find all dataset directories containing dataset.yaml."""
    ds_dir = root / 'datasets'
    if not ds_dir.exists():
        return []
    return sorted(d.name for d in ds_dir.iterdir() if d.is_dir() and (d / 'dataset.yaml').exists())
