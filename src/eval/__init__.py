"""
Ocean Health Monitoring Evaluation

This package provides comprehensive evaluation metrics and analysis tools
for ocean health monitoring models.
"""

from .ocean_evaluation import (
    OceanHealthEvaluator,
    evaluate_ocean_models
)

__all__ = [
    'OceanHealthEvaluator',
    'evaluate_ocean_models'
]
