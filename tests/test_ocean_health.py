"""
Test suite for Ocean Health Monitoring system.

This module contains unit tests for the core functionality of the ocean health
monitoring system, including data generation, model training, and evaluation.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.data.ocean_data import OceanDataGenerator, OceanHealthConfig, prepare_features
from src.models.ocean_models import ModelTrainer, ModelConfig, OceanHealthClassifier
from src.eval.ocean_evaluation import OceanHealthEvaluator


class TestOceanDataGenerator:
    """Test cases for ocean data generation."""
    
    def test_config_initialization(self):
        """Test configuration initialization."""
        config = OceanHealthConfig(n_samples=100, seed=42)
        assert config.n_samples == 100
        assert config.seed == 42
        assert config.n_regions == 50
    
    def test_data_generation(self):
        """Test synthetic data generation."""
        config = OceanHealthConfig(n_samples=100, seed=42)
        generator = OceanDataGenerator(config)
        data = generator.generate_dataset()
        
        # Check data shape
        assert len(data) == 100
        assert len(data.columns) > 10  # Should have multiple columns
        
        # Check required columns
        required_cols = [
            'longitude', 'latitude', 'date', 'sea_surface_temperature',
            'chlorophyll_concentration', 'ph_level', 'dissolved_oxygen',
            'salinity', 'health_status', 'health_label'
        ]
        for col in required_cols:
            assert col in data.columns
    
    def test_health_labels(self):
        """Test health label generation."""
        config = OceanHealthConfig(n_samples=100, seed=42)
        generator = OceanDataGenerator(config)
        data = generator.generate_dataset()
        
        # Check health status values
        assert set(data['health_status'].unique()).issubset({0, 1, 2})
        assert set(data['health_label'].unique()).issubset({'Healthy', 'Moderate Risk', 'Critical'})
        
        # Check that all samples have labels
        assert data['health_status'].notna().all()
        assert data['health_label'].notna().all()
    
    def test_spatial_features(self):
        """Test spatial feature generation."""
        config = OceanHealthConfig(n_samples=100, seed=42)
        generator = OceanDataGenerator(config)
        data = generator.generate_dataset()
        
        # Check spatial features
        assert 'ocean_basin' in data.columns
        assert 'distance_from_equator' in data.columns
        assert 'season' in data.columns
        
        # Check coordinate ranges
        assert data['longitude'].min() >= -180
        assert data['longitude'].max() <= 180
        assert data['latitude'].min() >= -90
        assert data['latitude'].max() <= 90


class TestModelTrainer:
    """Test cases for model training."""
    
    def test_model_config(self):
        """Test model configuration."""
        config = ModelConfig(random_state=42, test_size=0.2)
        assert config.random_state == 42
        assert config.test_size == 0.2
        assert config.cv_folds == 5
    
    def test_baseline_models(self):
        """Test baseline model creation."""
        config = ModelConfig(random_state=42)
        trainer = ModelTrainer(config)
        models = trainer.get_baseline_models()
        
        # Check that we have multiple models
        assert len(models) >= 4  # At least 4 baseline models
        
        # Check specific models
        expected_models = ['logistic_regression', 'random_forest', 'gradient_boosting']
        for model_name in expected_models:
            assert model_name in models
    
    def test_neural_network(self):
        """Test neural network model."""
        model = OceanHealthClassifier(input_dim=5, hidden_dims=[32], num_classes=3)
        
        # Test forward pass
        import torch
        x = torch.randn(10, 5)
        output = model(x)
        
        assert output.shape == (10, 3)  # Batch size 10, 3 classes


class TestOceanHealthEvaluator:
    """Test cases for evaluation metrics."""
    
    def test_evaluator_initialization(self):
        """Test evaluator initialization."""
        evaluator = OceanHealthEvaluator()
        assert evaluator.class_names == ['Healthy', 'Moderate Risk', 'Critical']
        assert len(evaluator.class_colors) == 3
    
    def test_classification_evaluation(self):
        """Test classification evaluation."""
        evaluator = OceanHealthEvaluator()
        
        # Create dummy data
        y_true = np.array([0, 1, 2, 0, 1])
        y_pred = np.array([0, 1, 1, 0, 2])
        y_pred_proba = np.array([[0.8, 0.1, 0.1],
                                [0.1, 0.7, 0.2],
                                [0.2, 0.6, 0.2],
                                [0.9, 0.05, 0.05],
                                [0.1, 0.2, 0.7]])
        
        results = evaluator.evaluate_classification(y_true, y_pred, y_pred_proba, "test_model")
        
        # Check that results contain expected keys
        expected_keys = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
        for key in expected_keys:
            assert key in results
        
        # Check that accuracy is a valid value
        assert 0 <= results['accuracy'] <= 1


class TestIntegration:
    """Integration tests for the full pipeline."""
    
    def test_full_pipeline(self):
        """Test the complete pipeline from data generation to evaluation."""
        # Generate data
        config = OceanHealthConfig(n_samples=200, seed=42)
        generator = OceanDataGenerator(config)
        data = generator.generate_dataset()
        
        # Prepare features
        X, y = prepare_features(data)
        assert X.shape[0] == len(data)
        assert len(y) == len(data)
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train models
        model_config = ModelConfig(random_state=42)
        trainer = ModelTrainer(model_config)
        
        # Train at least one baseline model
        models = trainer.get_baseline_models()
        model_name = list(models.keys())[0]
        model = models[model_name]
        
        # Create pipeline with scaling
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', model)
        ])
        
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        
        # Evaluate
        evaluator = OceanHealthEvaluator()
        results = evaluator.evaluate_classification(y_test, predictions, model_name="test")
        
        # Check that evaluation completed successfully
        assert 'accuracy' in results
        assert 0 <= results['accuracy'] <= 1


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
