"""Tests for the transform registry."""

from pathlib import Path

import pytest

from engram.transforms import resolve_transform, validate_transform_name


def test_validate_accepts_module_function_form():
    validate_transform_name('transforms.shape_input')
    validate_transform_name('pkg.sub.func')


def test_validate_rejects_names_without_dot():
    with pytest.raises(ValueError, match=r'module\.function'):
        validate_transform_name('just_a_name')


def test_resolve_loads_callable_from_impl_dir(tmp_path: Path):
    (tmp_path / 'transforms.py').write_text('def double(x):\n    return x * 2\n')

    fn = resolve_transform('transforms.double', tmp_path)
    assert fn(3) == 6


def test_resolve_raises_when_module_missing(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        resolve_transform('transforms.any', tmp_path)


def test_resolve_raises_when_function_missing(tmp_path: Path):
    (tmp_path / 'transforms.py').write_text('def exists(x):\n    return x\n')

    with pytest.raises(ValueError, match='missing_func'):
        resolve_transform('transforms.missing_func', tmp_path)
