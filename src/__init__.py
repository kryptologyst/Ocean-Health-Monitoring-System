"""
Ocean Health Monitoring System

A comprehensive ocean health monitoring system that combines machine learning models
with spatial analysis to assess marine ecosystem health.
"""

__version__ = "1.0.0"
__author__ = "kryptologyst"
__email__ = "kryptologyst@example.com"
__github__ = "https://github.com/kryptologyst"

# Import main components
from .data import OceanDataGenerator, OceanHealthConfig
from .models import ModelTrainer, OceanHealthClassifier
from .eval import OceanHealthEvaluator
from .viz import OceanHealthVisualizer

__all__ = [
    'OceanDataGenerator',
    'OceanHealthConfig',
    'ModelTrainer', 
    'OceanHealthClassifier',
    'OceanHealthEvaluator',
    'OceanHealthVisualizer'
]
