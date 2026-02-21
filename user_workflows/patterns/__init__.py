"""Pattern package for SLM analytical generators."""

from user_workflows.patterns.base import BasePattern, PatternResult, get_pattern

# Import modules for side-effect registration.
from user_workflows.patterns import (  # noqa: F401
    double_gaussian,
    gaussian_lattice,
    laguerre_gaussian,
    single_gaussian,
)

__all__ = ["BasePattern", "PatternResult", "get_pattern"]
