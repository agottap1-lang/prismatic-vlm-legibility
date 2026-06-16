"""
__init__.py

Probability-based Prismatic legibility evaluation.
"""

from .client import PrismaticTemporalClient
from .frames import create_frame_montage, extract_frame_sequence
from .prompt import (
    build_probability_prompt,
    parse_probability_response,
    calculate_decision_metrics,
)

__all__ = [
    "PrismaticTemporalClient",
    "create_frame_montage",
    "extract_frame_sequence",
    "build_probability_prompt",
    "parse_probability_response",
    "calculate_decision_metrics",
]
