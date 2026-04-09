#!/usr/bin/env python3
"""
Ocean Health Monitoring - Quick Start Example

This script demonstrates the basic usage of the Ocean Health Monitoring system.
It generates sample data, trains models, evaluates performance, and creates visualizations.
"""

import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.data.ocean_data import OceanDataGenerator, OceanHealthConfig, prepare_features
from src.models.ocean_models import train_ocean_health_models, ModelConfig
from src.eval.ocean_evaluation import evaluate_ocean_models
from src.viz.ocean_visualization import OceanHealthVisualizer


def main():
    """Run a complete ocean health monitoring example."""
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    logger.info("🌊 Starting Ocean Health Monitoring Example")
    
    # Step 1: Generate synthetic ocean health data
    logger.info("Step 1: Generating ocean health data...")
    config = OceanHealthConfig(n_samples=2000, seed=42)
    generator = OceanDataGenerator(config)
    data = generator.generate_dataset()
    
    logger.info(f"Generated {len(data)} ocean health samples")
    logger.info(f"Health status distribution:\n{data['health_label'].value_counts()}")
    
    # Step 2: Prepare features for machine learning
    logger.info("Step 2: Preparing features for ML...")
    X, y = prepare_features(data)
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    logger.info(f"Training set: {X_train.shape[0]} samples")
    logger.info(f"Test set: {X_test.shape[0]} samples")
    
    # Step 3: Train multiple models
    logger.info("Step 3: Training ocean health models...")
    model_config = ModelConfig(random_state=42)
    trainer = train_ocean_health_models(X_train, y_train, X_test, y_test, model_config)
    
    # Step 4: Evaluate models
    logger.info("Step 4: Evaluating model performance...")
    evaluator = evaluate_ocean_models(trainer, X_test, y_test, data)
    
    # Display results
    leaderboard = evaluator.create_evaluation_report(evaluator.results)
    print("\n🏆 Model Performance Leaderboard:")
    print("=" * 60)
    print(leaderboard.to_string(index=False))
    
    # Step 5: Create visualizations
    logger.info("Step 5: Creating visualizations...")
    visualizer = OceanHealthVisualizer()
    
    # Create output directory
    output_dir = Path("example_output")
    output_dir.mkdir(exist_ok=True)
    
    # Save visualizations
    visualizer.save_all_visualizations(
        data, 
        model_name="Example_Run",
        output_dir=str(output_dir)
    )
    
    logger.info(f"Visualizations saved to {output_dir}/")
    
    # Step 6: Display key insights
    logger.info("Step 6: Key Insights...")
    
    # Best model
    best_model_name = leaderboard.iloc[0]['Model']
    best_accuracy = leaderboard.iloc[0]['Test Accuracy']
    
    print(f"\n🎯 Best Model: {best_model_name}")
    print(f"📊 Best Accuracy: {best_accuracy:.4f}")
    
    # Health status summary
    health_summary = data['health_label'].value_counts()
    print(f"\n🌊 Ocean Health Summary:")
    for status, count in health_summary.items():
        percentage = (count / len(data)) * 100
        print(f"  {status}: {count} regions ({percentage:.1f}%)")
    
    # Critical regions analysis
    critical_data = data[data['health_label'] == 'Critical']
    if len(critical_data) > 0:
        avg_temp = critical_data['sea_surface_temperature'].mean()
        avg_oxygen = critical_data['dissolved_oxygen'].mean()
        print(f"\n⚠️  Critical Regions Analysis:")
        print(f"  Average Temperature: {avg_temp:.1f}°C")
        print(f"  Average Oxygen: {avg_oxygen:.1f} mg/L")
        print(f"  Count: {len(critical_data)} regions")
    
    logger.info("✅ Ocean Health Monitoring Example Completed Successfully!")
    logger.info("📁 Check the 'example_output' directory for generated visualizations")
    logger.info("🚀 Run 'streamlit run demo/ocean_health_demo.py' for interactive dashboard")


if __name__ == "__main__":
    main()
