"""
Ocean Health Monitoring Evaluation

This module provides comprehensive evaluation metrics and analysis tools
for ocean health monitoring models, including domain-specific metrics.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import logging
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, roc_curve
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
import warnings

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


class OceanHealthEvaluator:
    """Comprehensive evaluator for ocean health monitoring models."""
    
    def __init__(self, class_names: List[str] = None):
        self.class_names = class_names or ['Healthy', 'Moderate Risk', 'Critical']
        self.results = {}
        
    def evaluate_classification(self, y_true: np.ndarray, y_pred: np.ndarray, 
                              y_pred_proba: Optional[np.ndarray] = None,
                              model_name: str = "model") -> Dict[str, Any]:
        """Evaluate classification performance with comprehensive metrics."""
        
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        
        # Per-class metrics
        precision_per_class = precision_score(y_true, y_pred, average=None)
        recall_per_class = recall_score(y_true, y_pred, average=None)
        f1_per_class = f1_score(y_true, y_pred, average=None)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # ROC AUC (if probabilities available)
        roc_auc = None
        if y_pred_proba is not None:
            try:
                roc_auc = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='weighted')
            except ValueError:
                logger.warning("Could not calculate ROC AUC - check if probabilities are valid")
        
        results = {
            'model_name': model_name,
            'accuracy': accuracy,
            'precision_weighted': precision,
            'recall_weighted': recall,
            'f1_weighted': f1,
            'precision_per_class': precision_per_class,
            'recall_per_class': recall_per_class,
            'f1_per_class': f1_per_class,
            'confusion_matrix': cm,
            'roc_auc': roc_auc,
            'classification_report': classification_report(y_true, y_pred, target_names=self.class_names)
        }
        
        self.results[model_name] = results
        return results
    
    def evaluate_spatial_performance(self, data: pd.DataFrame, y_true: np.ndarray, 
                                  y_pred: np.ndarray, model_name: str = "model") -> Dict[str, Any]:
        """Evaluate spatial performance metrics for ocean health models."""
        
        # Add predictions to data
        eval_data = data.copy()
        eval_data['y_true'] = y_true
        eval_data['y_pred'] = y_pred
        eval_data['correct'] = (y_true == y_pred)
        
        # Spatial accuracy by ocean basin
        basin_accuracy = eval_data.groupby('ocean_basin')['correct'].mean()
        
        # Spatial accuracy by latitude bands
        eval_data['lat_band'] = pd.cut(eval_data['latitude'], bins=5, labels=['S5', 'S4', 'S3', 'S2', 'S1'])
        lat_accuracy = eval_data.groupby('lat_band')['correct'].mean()
        
        # Spatial accuracy by longitude bands
        eval_data['lon_band'] = pd.cut(eval_data['longitude'], bins=5, labels=['W5', 'W4', 'W3', 'W2', 'W1'])
        lon_accuracy = eval_data.groupby('lon_band')['correct'].mean()
        
        # Seasonal performance
        seasonal_accuracy = eval_data.groupby('season')['correct'].mean()
        
        spatial_results = {
            'model_name': model_name,
            'basin_accuracy': basin_accuracy,
            'latitude_accuracy': lat_accuracy,
            'longitude_accuracy': lon_accuracy,
            'seasonal_accuracy': seasonal_accuracy,
            'spatial_variance': eval_data['correct'].var()
        }
        
        return spatial_results
    
    def evaluate_domain_metrics(self, data: pd.DataFrame, y_true: np.ndarray, 
                              y_pred: np.ndarray, model_name: str = "model") -> Dict[str, Any]:
        """Evaluate domain-specific metrics for ocean health monitoring."""
        
        eval_data = data.copy()
        eval_data['y_true'] = y_true
        eval_data['y_pred'] = y_pred
        
        # Critical condition detection (most important for ocean health)
        critical_mask = y_true == 2
        critical_detected = (y_pred == 2) & critical_mask
        critical_recall = critical_detected.sum() / critical_mask.sum() if critical_mask.sum() > 0 else 0
        
        # False alarm rate for critical conditions
        false_critical = (y_pred == 2) & (y_true != 2)
        false_alarm_rate = false_critical.sum() / (y_true != 2).sum() if (y_true != 2).sum() > 0 else 0
        
        # Healthy condition preservation
        healthy_mask = y_true == 0
        healthy_preserved = (y_pred == 0) & healthy_mask
        healthy_precision = healthy_preserved.sum() / (y_pred == 0).sum() if (y_pred == 0).sum() > 0 else 0
        
        # Temperature-based performance (critical for coral bleaching)
        temp_critical = eval_data['sea_surface_temperature'] > 28.5
        temp_performance = eval_data[temp_critical]['correct'].mean() if temp_critical.sum() > 0 else 0
        
        # Oxygen-based performance (critical for marine life)
        oxygen_critical = eval_data['dissolved_oxygen'] < 4.0
        oxygen_performance = eval_data[oxygen_critical]['correct'].mean() if oxygen_critical.sum() > 0 else 0
        
        domain_results = {
            'model_name': model_name,
            'critical_recall': critical_recall,
            'false_alarm_rate': false_alarm_rate,
            'healthy_precision': healthy_precision,
            'temp_critical_performance': temp_performance,
            'oxygen_critical_performance': oxygen_performance
        }
        
        return domain_results
    
    def create_evaluation_report(self, model_results: Dict[str, Any]) -> pd.DataFrame:
        """Create comprehensive evaluation report."""
        report_data = []
        
        for model_name, results in model_results.items():
            report_data.append({
                'Model': model_name,
                'Accuracy': results['accuracy'],
                'Precision (Weighted)': results['precision_weighted'],
                'Recall (Weighted)': results['recall_weighted'],
                'F1 (Weighted)': results['f1_weighted'],
                'ROC AUC': results['roc_auc'] if results['roc_auc'] is not None else 'N/A'
            })
        
        report_df = pd.DataFrame(report_data)
        report_df = report_df.sort_values('Accuracy', ascending=False)
        
        return report_df
    
    def plot_confusion_matrices(self, model_results: Dict[str, Any], 
                              save_path: Optional[str] = None):
        """Plot confusion matrices for all models."""
        n_models = len(model_results)
        fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4))
        
        if n_models == 1:
            axes = [axes]
        
        for i, (model_name, results) in enumerate(model_results.items()):
            cm = results['confusion_matrix']
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=self.class_names,
                       yticklabels=self.class_names,
                       ax=axes[i])
            axes[i].set_title(f'{model_name} Confusion Matrix')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrices saved to {save_path}")
        
        plt.show()
    
    def plot_per_class_metrics(self, model_results: Dict[str, Any],
                              save_path: Optional[str] = None):
        """Plot per-class precision, recall, and F1 scores."""
        metrics = ['precision_per_class', 'recall_per_class', 'f1_per_class']
        metric_names = ['Precision', 'Recall', 'F1 Score']
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for i, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
            data = []
            models = []
            
            for model_name, results in model_results.items():
                data.append(results[metric])
                models.append(model_name)
            
            data = np.array(data)
            
            x = np.arange(len(self.class_names))
            width = 0.8 / len(models)
            
            for j, model in enumerate(models):
                axes[i].bar(x + j * width, data[j], width, label=model)
            
            axes[i].set_xlabel('Ocean Health Class')
            axes[i].set_ylabel(metric_name)
            axes[i].set_title(f'{metric_name} by Class')
            axes[i].set_xticks(x + width * (len(models) - 1) / 2)
            axes[i].set_xticklabels(self.class_names)
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Per-class metrics saved to {save_path}")
        
        plt.show()
    
    def plot_spatial_performance(self, spatial_results: Dict[str, Any],
                               save_path: Optional[str] = None):
        """Plot spatial performance metrics."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Basin accuracy
        basin_data = spatial_results['basin_accuracy']
        axes[0, 0].bar(basin_data.index, basin_data.values)
        axes[0, 0].set_title('Accuracy by Ocean Basin')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Latitude accuracy
        lat_data = spatial_results['latitude_accuracy']
        axes[0, 1].bar(lat_data.index, lat_data.values)
        axes[0, 1].set_title('Accuracy by Latitude Band')
        axes[0, 1].set_ylabel('Accuracy')
        
        # Longitude accuracy
        lon_data = spatial_results['longitude_accuracy']
        axes[1, 0].bar(lon_data.index, lon_data.values)
        axes[1, 0].set_title('Accuracy by Longitude Band')
        axes[1, 0].set_ylabel('Accuracy')
        
        # Seasonal accuracy
        seasonal_data = spatial_results['seasonal_accuracy']
        axes[1, 1].bar(seasonal_data.index, seasonal_data.values)
        axes[1, 1].set_title('Accuracy by Season')
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Spatial performance plots saved to {save_path}")
        
        plt.show()


def evaluate_ocean_models(trainer, X_test: np.ndarray, y_test: np.ndarray,
                         data: pd.DataFrame) -> OceanHealthEvaluator:
    """Comprehensive evaluation of ocean health models."""
    
    evaluator = OceanHealthEvaluator()
    
    # Evaluate each model
    for model_name, results in trainer.results.items():
        y_pred = results['predictions']
        y_pred_proba = results.get('probabilities')
        
        # Classification evaluation
        eval_results = evaluator.evaluate_classification(
            y_test, y_pred, y_pred_proba, model_name
        )
        
        # Spatial evaluation
        spatial_results = evaluator.evaluate_spatial_performance(
            data, y_test, y_pred, model_name
        )
        
        # Domain-specific evaluation
        domain_results = evaluator.evaluate_domain_metrics(
            data, y_test, y_pred, model_name
        )
        
        # Combine results
        eval_results.update(spatial_results)
        eval_results.update(domain_results)
    
    return evaluator


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    from src.data.ocean_data import OceanDataGenerator, OceanHealthConfig, prepare_features
    from src.models.ocean_models import train_ocean_health_models
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
    
    # Evaluate models
    evaluator = evaluate_ocean_models(trainer, X_test, y_test, data)
    
    # Create evaluation report
    report = evaluator.create_evaluation_report(evaluator.results)
    print("\nEvaluation Report:")
    print(report.to_string(index=False))
    
    # Plot results
    evaluator.plot_confusion_matrices(evaluator.results)
    evaluator.plot_per_class_metrics(evaluator.results)
