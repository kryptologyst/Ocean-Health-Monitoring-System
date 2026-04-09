#!/usr/bin/env python3
"""
Simple Ocean Health Monitoring Test

This script tests the basic functionality without requiring heavy dependencies
like geopandas, folium, etc. It focuses on core ML functionality.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def generate_simple_ocean_data(n_samples=1000, seed=42):
    """Generate simple ocean health data without heavy dependencies."""
    np.random.seed(seed)
    
    # Generate ocean parameters
    sea_temp = np.random.normal(26.0, 2.5, n_samples)
    chlorophyll = np.random.lognormal(0.3, 0.8, n_samples)
    ph_level = np.random.normal(8.1, 0.15, n_samples)
    dissolved_oxygen = np.random.normal(6.0, 1.2, n_samples)
    salinity = np.random.normal(35.0, 1.5, n_samples)
    
    # Generate spatial coordinates
    longitude = np.random.uniform(-180, 180, n_samples)
    latitude = np.random.uniform(-90, 90, n_samples)
    
    # Generate temporal features
    dates = pd.date_range('2020-01-01', periods=n_samples, freq='D')
    
    # Create health labels based on thresholds
    critical_conditions = (
        (sea_temp > 28.5) | 
        (dissolved_oxygen < 4.0) | 
        (chlorophyll > 3.0) |
        (ph_level < 7.8) | (ph_level > 8.3)
    )
    
    moderate_conditions = (
        ((sea_temp > 27.0) | 
         (dissolved_oxygen < 5.5) | 
         (chlorophyll > 2.0) |
         (ph_level < 8.0) | (ph_level > 8.2)) & 
        ~critical_conditions
    )
    
    health_status = np.where(critical_conditions, 2,
                            np.where(moderate_conditions, 1, 0))
    
    health_labels = np.where(health_status == 0, 'Healthy',
                           np.where(health_status == 1, 'Moderate Risk', 'Critical'))
    
    # Create DataFrame
    data = pd.DataFrame({
        'longitude': longitude,
        'latitude': latitude,
        'date': dates,
        'sea_surface_temperature': sea_temp,
        'chlorophyll_concentration': chlorophyll,
        'ph_level': ph_level,
        'dissolved_oxygen': dissolved_oxygen,
        'salinity': salinity,
        'health_status': health_status,
        'health_label': health_labels
    })
    
    return data


def train_simple_model(data):
    """Train a simple Random Forest model."""
    # Prepare features
    feature_cols = [
        'sea_surface_temperature',
        'chlorophyll_concentration',
        'ph_level',
        'dissolved_oxygen',
        'salinity'
    ]
    
    X = data[feature_cols].values
    y = data['health_status'].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, X_test, y_test, y_pred, accuracy


def create_simple_visualizations(data, accuracy):
    """Create simple visualizations."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Health status distribution
    health_counts = data['health_label'].value_counts()
    axes[0, 0].bar(health_counts.index, health_counts.values, 
                   color=['green', 'orange', 'red'])
    axes[0, 0].set_title('Ocean Health Status Distribution')
    axes[0, 0].set_ylabel('Count')
    
    # Temperature distribution by health status
    for status in ['Healthy', 'Moderate Risk', 'Critical']:
        subset = data[data['health_label'] == status]
        axes[0, 1].hist(subset['sea_surface_temperature'], alpha=0.7, 
                       label=status, bins=20)
    axes[0, 1].set_title('Temperature Distribution by Health Status')
    axes[0, 1].set_xlabel('Temperature (°C)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].legend()
    
    # Oxygen distribution by health status
    for status in ['Healthy', 'Moderate Risk', 'Critical']:
        subset = data[data['health_label'] == status]
        axes[1, 0].hist(subset['dissolved_oxygen'], alpha=0.7, 
                       label=status, bins=20)
    axes[1, 0].set_title('Oxygen Distribution by Health Status')
    axes[1, 0].set_xlabel('Dissolved Oxygen (mg/L)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].legend()
    
    # Spatial distribution
    colors = {'Healthy': 'green', 'Moderate Risk': 'orange', 'Critical': 'red'}
    for status in ['Healthy', 'Moderate Risk', 'Critical']:
        subset = data[data['health_label'] == status]
        axes[1, 1].scatter(subset['longitude'], subset['latitude'], 
                          c=colors[status], label=status, alpha=0.6, s=10)
    axes[1, 1].set_title('Spatial Distribution of Ocean Health')
    axes[1, 1].set_xlabel('Longitude')
    axes[1, 1].set_ylabel('Latitude')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('simple_ocean_health_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig


def main():
    """Run simple ocean health monitoring example."""
    logger.info("🌊 Starting Simple Ocean Health Monitoring Example")
    
    # Generate data
    logger.info("Generating ocean health data...")
    data = generate_simple_ocean_data(n_samples=2000)
    
    logger.info(f"Generated {len(data)} ocean health samples")
    logger.info(f"Health status distribution:\n{data['health_label'].value_counts()}")
    
    # Train model
    logger.info("Training Random Forest model...")
    model, X_test, y_test, y_pred, accuracy = train_simple_model(data)
    
    logger.info(f"Model accuracy: {accuracy:.4f}")
    
    # Display classification report
    print("\n📊 Classification Report:")
    print("=" * 50)
    print(classification_report(y_test, y_pred, 
                              target_names=['Healthy', 'Moderate Risk', 'Critical']))
    
    # Create visualizations
    logger.info("Creating visualizations...")
    fig = create_simple_visualizations(data, accuracy)
    
    # Display key insights
    print(f"\n🎯 Model Performance:")
    print(f"   Accuracy: {accuracy:.4f}")
    
    health_summary = data['health_label'].value_counts()
    print(f"\n🌊 Ocean Health Summary:")
    for status, count in health_summary.items():
        percentage = (count / len(data)) * 100
        print(f"   {status}: {count} regions ({percentage:.1f}%)")
    
    # Critical regions analysis
    critical_data = data[data['health_label'] == 'Critical']
    if len(critical_data) > 0:
        avg_temp = critical_data['sea_surface_temperature'].mean()
        avg_oxygen = critical_data['dissolved_oxygen'].mean()
        print(f"\n⚠️  Critical Regions Analysis:")
        print(f"   Average Temperature: {avg_temp:.1f}°C")
        print(f"   Average Oxygen: {avg_oxygen:.1f} mg/L")
        print(f"   Count: {len(critical_data)} regions")
    
    logger.info("✅ Simple Ocean Health Monitoring Example Completed!")
    logger.info("📁 Check 'simple_ocean_health_analysis.png' for visualizations")


if __name__ == "__main__":
    main()
