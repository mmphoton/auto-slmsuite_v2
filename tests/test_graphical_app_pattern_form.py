import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from user_workflows.graphical_app.app.patterns import PatternService
from user_workflows.graphical_app.ui.pattern_form import parity_check_for_schema


REQUIRED_METADATA_KEYS = {"name", "type", "default", "range", "options", "help"}


def test_schema_metadata_exposes_full_parameter_details():
    service = PatternService()
    for pattern in service.available_patterns():
        schema = service.schema_for(pattern)
        assert schema["pattern"] == pattern
        assert schema["parameters"]
        for parameter in schema["parameters"]:
            assert REQUIRED_METADATA_KEYS.issubset(parameter.keys())


def test_pattern_form_parity_covers_all_schema_parameters():
    service = PatternService()
    for pattern in service.available_patterns():
        schema = service.schema_for(pattern)
        represented = {param["name"] for param in schema["parameters"]}
        assert parity_check_for_schema(schema, represented) == []
