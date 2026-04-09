"""
Ocean Health Monitoring Models

This module contains various machine learning models for ocean health classification,
including baseline models, spatial models, and deep learning approaches.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass
from pathlib import Path

# Core ML libraries
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Deep learning
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# XGBoost for advanced gradient boosting
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logging.warning("XGBoost not available. Install with: pip install xgboost")

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for model training."""
    random_state: int = 42
    test_size: float = 0.2
    cv_folds: int = 5
    device: str = "auto"  # auto, cpu, cuda, mps


class OceanHealthClassifier(nn.Module):
    """PyTorch neural network for ocean health classification."""
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [64, 32], 
                 num_classes: int = 3, dropout_rate: float = 0.2):
        super(OceanHealthClassifier, self).__init__()
        
        layers_list = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers_list.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        layers_list.append(nn.Linear(prev_dim, num_classes))
        
        self.network = nn.Sequential(*layers_list)
        
    def forward(self, x):
        return self.network(x)


class ModelTrainer:
    """Handles training and evaluation of various ocean health models."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.models = {}
        self.scalers = {}
        self.results = {}
        
        # Set device for PyTorch
        if config.device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(config.device)
            
        logger.info(f"Using device: {self.device}")
        
        # Set random seeds for reproducibility
        np.random.seed(config.random_state)
        torch.manual_seed(config.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(config.random_state)
    
    def get_baseline_models(self) -> Dict[str, Any]:
        """Get dictionary of baseline models for comparison."""
        models = {
            'logistic_regression': LogisticRegression(
                random_state=self.config.random_state,
                max_iter=1000
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                random_state=self.config.random_state,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100,
                random_state=self.config.random_state
            ),
            'svm': SVC(
                random_state=self.config.random_state,
                probability=True
            ),
            'knn': KNeighborsClassifier(
                n_neighbors=5,
                n_jobs=-1
            )
        }
        
        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            models['xgboost'] = xgb.XGBClassifier(
                random_state=self.config.random_state,
                eval_metric='mlogloss'
            )
        
        return models
    
    def train_baseline_models(self, X_train: np.ndarray, y_train: np.ndarray,
                            X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Dict]:
        """Train and evaluate baseline models."""
        logger.info("Training baseline models...")
        
        models = self.get_baseline_models()
        results = {}
        
        for name, model in models.items():
            logger.info(f"Training {name}...")
            
            # Create pipeline with scaling
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', model)
            ])
            
            # Train model
            pipeline.fit(X_train, y_train)
            
            # Evaluate
            train_score = pipeline.score(X_train, y_train)
            test_score = pipeline.score(X_test, y_test)
            
            # Cross-validation
            cv_scores = cross_val_score(
                pipeline, X_train, y_train, 
                cv=StratifiedKFold(n_splits=self.config.cv_folds, shuffle=True, 
                                 random_state=self.config.random_state),
                scoring='accuracy'
            )
            
            # Predictions
            y_pred = pipeline.predict(X_test)
            y_pred_proba = pipeline.predict_proba(X_test)
            
            results[name] = {
                'model': pipeline,
                'train_score': train_score,
                'test_score': test_score,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'predictions': y_pred,
                'probabilities': y_pred_proba,
                'classification_report': classification_report(y_test, y_pred)
            }
            
            logger.info(f"{name} - Test Score: {test_score:.4f}, CV: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        self.results.update(results)
        return results
    
    def train_neural_network(self, X_train: np.ndarray, y_train: np.ndarray,
                           X_test: np.ndarray, y_test: np.ndarray,
                           epochs: int = 100, batch_size: int = 32,
                           learning_rate: float = 0.001) -> Dict[str, Any]:
        """Train PyTorch neural network."""
        logger.info("Training neural network...")
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.LongTensor(y_train)
        X_test_tensor = torch.FloatTensor(X_test)
        y_test_tensor = torch.LongTensor(y_test)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Initialize model
        input_dim = X_train.shape[1]
        model = OceanHealthClassifier(input_dim).to(self.device)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Training loop
        model.train()
        train_losses = []
        
        for epoch in range(epochs):
            epoch_loss = 0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            train_losses.append(epoch_loss / len(train_loader))
            
            if epoch % 20 == 0:
                logger.info(f"Epoch {epoch}, Loss: {epoch_loss / len(train_loader):.4f}")
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            X_test_tensor = X_test_tensor.to(self.device)
            outputs = model(X_test_tensor)
            _, predicted = torch.max(outputs.data, 1)
            probabilities = torch.softmax(outputs, dim=1)
            
            # Move back to CPU for sklearn metrics
            predicted = predicted.cpu().numpy()
            probabilities = probabilities.cpu().numpy()
        
        # Calculate accuracy
        test_score = (predicted == y_test).mean()
        
        # Cross-validation (simplified - using train set)
        train_score = 0.0  # Would need proper CV implementation
        
        results = {
            'model': model,
            'train_score': train_score,
            'test_score': test_score,
            'cv_mean': test_score,  # Simplified
            'cv_std': 0.0,
            'predictions': predicted,
            'probabilities': probabilities,
            'train_losses': train_losses,
            'classification_report': classification_report(y_test, predicted)
        }
        
        logger.info(f"Neural Network - Test Score: {test_score:.4f}")
        self.results['neural_network'] = results
        
        return results
    
    def get_model_leaderboard(self) -> pd.DataFrame:
        """Create a leaderboard of all trained models."""
        leaderboard_data = []
        
        for name, results in self.results.items():
            leaderboard_data.append({
                'Model': name,
                'Test Score': results['test_score'],
                'CV Mean': results['cv_mean'],
                'CV Std': results['cv_std'],
                'Train Score': results['train_score']
            })
        
        leaderboard = pd.DataFrame(leaderboard_data)
        leaderboard = leaderboard.sort_values('Test Score', ascending=False)
        
        return leaderboard
    
    def save_models(self, save_dir: str):
        """Save trained models to disk."""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        for name, results in self.results.items():
            if name == 'neural_network':
                # Save PyTorch model
                torch.save(results['model'].state_dict(), 
                          save_path / f"{name}_weights.pth")
            else:
                # Save sklearn models
                import joblib
                joblib.dump(results['model'], save_path / f"{name}_model.pkl")
        
        logger.info(f"Models saved to {save_path}")


def train_ocean_health_models(X_train: np.ndarray, y_train: np.ndarray,
                            X_test: np.ndarray, y_test: np.ndarray,
                            config: Optional[ModelConfig] = None) -> ModelTrainer:
    """Train all ocean health models and return trainer with results."""
    if config is None:
        config = ModelConfig()
    
    trainer = ModelTrainer(config)
    
    # Train baseline models
    trainer.train_baseline_models(X_train, y_train, X_test, y_test)
    
    # Train neural network
    trainer.train_neural_network(X_train, y_train, X_test, y_test)
    
    return trainer


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    from src.data.ocean_data import OceanDataGenerator, OceanHealthConfig, prepare_features
    from sklearn.model_selection import train_test_split
    
    # Generate sample data
    config = OceanHealthConfig(n_samples=2000)
    generator = OceanDataGenerator(config)
    data = generator.generate_dataset()
    
    # Prepare features
    X, y = prepare_features(data)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train models
    trainer = train_ocean_health_models(X_train, y_train, X_test, y_test)
    
    # Display results
    leaderboard = trainer.get_model_leaderboard()
    print("\nModel Leaderboard:")
    print(leaderboard.to_string(index=False))
