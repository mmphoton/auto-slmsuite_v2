import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture
def pattern_registry_module():
    return pytest.importorskip("user_workflows.patterns.registry")


@pytest.fixture
def clean_registry(pattern_registry_module):
    registry_cls = getattr(pattern_registry_module, "PatternRegistry")
    return registry_cls()


def _call_first(obj, names, *args, **kwargs):
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)(*args, **kwargs)
    raise AttributeError(f"None of {names} were found on {obj!r}")


def _list_registered(registry):
    result = _call_first(registry, ["list", "list_patterns", "names", "registered"])
    if isinstance(result, dict):
        return list(result)
    return list(result)


def _lookup(registry, name):
    return _call_first(registry, ["lookup", "get"], name)


def _register(registry, name, builder, schema):
    return _call_first(
        registry,
        ["register", "add", "register_pattern"],
        name=name,
        builder=builder,
        schema=schema,
    )


def test_registry_registration_lookup_and_listing(clean_registry):
    def builder(params, context):
        return np.zeros(context.shape, dtype=float)

    _register(
        clean_registry,
        name="unit-test-pattern",
        builder=builder,
        schema={"type": "object", "properties": {}, "additionalProperties": False},
    )

    names = _list_registered(clean_registry)
    assert "unit-test-pattern" in names

    entry = _lookup(clean_registry, "unit-test-pattern")
    assert entry is not None


def test_duplicate_registration_fails(clean_registry):
    def builder(params, context):
        return np.zeros(context.shape, dtype=float)

    schema = {"type": "object", "properties": {}, "additionalProperties": False}
    _register(clean_registry, name="dup", builder=builder, schema=schema)

    with pytest.raises((ValueError, KeyError, RuntimeError)):
        _register(clean_registry, name="dup", builder=builder, schema=schema)


def test_schema_validation_success_and_failure(clean_registry):
    def builder(params, context):
        return np.zeros(context.shape, dtype=float)

    schema = {
        "type": "object",
        "properties": {
            "amplitude": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        },
        "required": ["amplitude"],
        "additionalProperties": False,
    }
    _register(clean_registry, name="schema-pattern", builder=builder, schema=schema)

    validate = lambda payload: _call_first(
        clean_registry,
        ["validate", "validate_params", "validate_pattern_params"],
        "schema-pattern",
        payload,
    )

    validate({"amplitude": 0.5})
    with pytest.raises(Exception):
        validate({"amplitude": 1.5})
    with pytest.raises(Exception):
        validate({"wrong": 0.5})


def test_each_registered_pattern_returns_expected_phase_shape_and_range(pattern_registry_module):
    registry = getattr(pattern_registry_module, "registry", None)
    if registry is None:
        registry = getattr(pattern_registry_module, "get_default_registry")()

    names = _list_registered(registry)
    assert names, "Expected at least one registered pattern."

    context = SimpleNamespace(shape=(16, 24), slm_shape=(16, 24))

    for name in names:
        params = {}
        if any(hasattr(registry, fn) for fn in ["default_params", "get_default_params"]):
            maybe_defaults = _call_first(registry, ["default_params", "get_default_params"], name)
            if isinstance(maybe_defaults, dict):
                params.update(maybe_defaults)

        phase = _call_first(registry, ["build", "generate", "create"], name, params, context)
        phase = np.asarray(phase)

        assert phase.shape == context.shape
        assert np.isfinite(phase).all()
        assert np.nanmin(phase) >= 0.0
        assert np.nanmax(phase) <= (2.0 * np.pi + 1e-9)


def test_two_pattern_compositing_behaves_mod_2pi():
    compositor = pytest.importorskip("user_workflows.patterns.compositor")

    context = SimpleNamespace(shape=(8, 8))
    a = np.full(context.shape, 1.75 * np.pi)
    b = np.full(context.shape, 1.25 * np.pi)

    compose = _call_first(compositor, ["compose", "compose_patterns", "composite"])
    result = compose([a, b], context=context)

    expected = np.mod(a + b, 2 * np.pi)
    assert np.allclose(np.asarray(result), expected)
