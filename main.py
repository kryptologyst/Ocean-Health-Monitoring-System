#!/usr/bin/env python3
"""
Ocean Health Monitoring - Main Script

This script provides a command-line interface for the ocean health monitoring system.
It can generate data, train models, evaluate performance, and create visualizations.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.data.ocean_data import OceanDataGenerator, OceanHealthConfig, prepare_features
from src.models.ocean_models import train_ocean_health_models, ModelConfig
from src.eval.ocean_evaluation import evaluate_ocean_models, OceanHealthEvaluator
from src.viz.ocean_visualization import OceanHealthVisualizer


def setup_logging(level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('ocean_health.log')
        ]
    )


def generate_data(args) -> None:
    """Generate synthetic ocean health data."""
    logging.info("Generating ocean health data...")
    
    config = OceanHealthConfig(
        n_samples=args.samples,
        seed=args.seed
    )
    
    generator = OceanDataGenerator(config)
    data = generator.generate_dataset()
    
    # Save data
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    data_file = output_path / "ocean_health_data.csv"
    data.to_csv(data_file, index=False)
    
    logging.info(f"Data saved to {data_file}")
    logging.info(f"Generated {len(data)} samples")
    logging.info(f"Health status distribution:\n{data['health_label'].value_counts()}")


def train_models(args) -> None:
    """Train ocean health models."""
    logging.info("Training ocean health models...")
    
    # Load or generate data
    data_path = Path(args.data_file) if args.data_file else None
    if data_path and data_path.exists():
        import pandas as pd
        data = pd.read_csv(data_path)
        logging.info(f"Loaded data from {data_path}")
    else:
        config = OceanHealthConfig(n_samples=args.samples, seed=args.seed)
        generator = OceanDataGenerator(config)
        data = generator.generate_dataset()
        logging.info("Generated synthetic data")
    
    # Prepare features
    X, y = prepare_features(data)
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )
    
    # Train models
    model_config = ModelConfig(
        random_state=args.seed,
        test_size=args.test_size,
        cv_folds=args.cv_folds
    )
    
    trainer = train_ocean_health_models(X_train, y_train, X_test, y_test, model_config)
    
    # Evaluate models
    evaluator = evaluate_ocean_models(trainer, X_test, y_test, data)
    
    # Display results
    leaderboard = evaluator.create_evaluation_report(evaluator.results)
    print("\nModel Performance Leaderboard:")
    print(leaderboard.to_string(index=False))
    
    # Save models
    if args.save_models:
        models_dir = Path(args.output_dir) / "models"
        trainer.save_models(str(models_dir))
        logging.info(f"Models saved to {models_dir}")


def create_visualizations(args) -> None:
    """Create visualizations for ocean health data."""
    logging.info("Creating visualizations...")
    
    # Load data
    data_path = Path(args.data_file) if args.data_file else None
    if data_path and data_path.exists():
        import pandas as pd
        data = pd.read_csv(data_path)
        logging.info(f"Loaded data from {data_path}")
    else:
        config = OceanHealthConfig(n_samples=args.samples, seed=args.seed)
        generator = OceanDataGenerator(config)
        data = generator.generate_dataset()
        logging.info("Generated synthetic data")
    
    # Create visualizer
    visualizer = OceanHealthVisualizer()
    
    # Create visualizations
    output_dir = Path(args.output_dir) / "visualizations"
    visualizer.save_all_visualizations(
        data, 
        model_name=args.model_name,
        output_dir=str(output_dir)
    )
    
    logging.info(f"Visualizations saved to {output_dir}")


def run_demo(args) -> None:
    """Run the Streamlit demo application."""
    import subprocess
    
    demo_path = Path(__file__).parent / "demo" / "ocean_health_demo.py"
    
    if not demo_path.exists():
        logging.error(f"Demo file not found: {demo_path}")
        return
    
    logging.info("Starting Streamlit demo...")
    subprocess.run([
        "streamlit", "run", str(demo_path),
        "--server.port", str(args.port),
        "--server.address", args.host
    ])


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(
        description="Ocean Health Monitoring System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate sample data
  python main.py generate --samples 2000 --output-dir data/
  
  # Train models
  python main.py train --samples 2000 --save-models
  
  # Create visualizations
  python main.py visualize --data-file data/ocean_health_data.csv
  
  # Run demo
  python main.py demo --port 8501
        """
    )
    
    # Global arguments
    parser.add_argument("--log-level", default="INFO", 
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Set logging level")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    
    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Generate data command
    gen_parser = subparsers.add_parser("generate", help="Generate synthetic ocean data")
    gen_parser.add_argument("--samples", type=int, default=2000,
                          help="Number of samples to generate")
    gen_parser.add_argument("--output-dir", default="data/",
                          help="Output directory for generated data")
    
    # Train models command
    train_parser = subparsers.add_parser("train", help="Train ocean health models")
    train_parser.add_argument("--data-file", 
                            help="Path to data file (generates if not provided)")
    train_parser.add_argument("--samples", type=int, default=2000,
                            help="Number of samples for training")
    train_parser.add_argument("--test-size", type=float, default=0.2,
                            help="Test set size")
    train_parser.add_argument("--cv-folds", type=int, default=5,
                            help="Cross-validation folds")
    train_parser.add_argument("--save-models", action="store_true",
                            help="Save trained models")
    train_parser.add_argument("--output-dir", default="output/",
                            help="Output directory")
    
    # Visualize command
    viz_parser = subparsers.add_parser("visualize", help="Create visualizations")
    viz_parser.add_argument("--data-file",
                          help="Path to data file (generates if not provided)")
    viz_parser.add_argument("--samples", type=int, default=2000,
                          help="Number of samples for visualization")
    viz_parser.add_argument("--model-name", default="OceanHealth",
                          help="Model name for visualization titles")
    viz_parser.add_argument("--output-dir", default="assets/",
                          help="Output directory for visualizations")
    
    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Run Streamlit demo")
    demo_parser.add_argument("--host", default="localhost",
                           help="Host for demo server")
    demo_parser.add_argument("--port", type=int, default=8501,
                           help="Port for demo server")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Execute command
    if args.command == "generate":
        generate_data(args)
    elif args.command == "train":
        train_models(args)
    elif args.command == "visualize":
        create_visualizations(args)
    elif args.command == "demo":
        run_demo(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
