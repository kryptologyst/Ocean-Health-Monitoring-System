"""
Ocean Health Monitoring Models

This package provides machine learning models for ocean health classification,
including baseline models, spatial models, and deep learning approaches.
"""

from .ocean_models import (
    OceanHealthClassifier,
    ModelTrainer,
    ModelConfig,
    train_ocean_health_models
)

__all__ = [
    'OceanHealthClassifier',
    'ModelTrainer',
    'ModelConfig',
    'train_ocean_health_models'
]
