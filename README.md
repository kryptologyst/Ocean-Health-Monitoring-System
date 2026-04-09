# Ocean Health Monitoring System

A comprehensive ocean health monitoring system that combines machine learning models with spatial analysis to assess marine ecosystem health. This system provides real-time analysis, interactive visualizations, and early warning capabilities for critical ocean conditions.

## Features

- **Multi-Model ML Pipeline**: Compare various algorithms including neural networks, ensemble methods, and traditional classifiers
- **Spatial Analysis**: Interactive maps and geographic visualization of ocean health status
- **Real-time Monitoring**: Process ocean health data with comprehensive parameter analysis
- **Domain-Specific Metrics**: Specialized evaluation metrics for marine ecosystem health
- **Interactive Dashboard**: Streamlit-based web application for exploration and analysis

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/kryptologyst/Ocean-Health-Monitoring-System.git
cd Ocean-Health-Monitoring-System

# Install dependencies
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

### Running the Demo

```bash
# Launch the interactive Streamlit dashboard
streamlit run demo/ocean_health_demo.py

# Or use the command line interface
ocean-health-demo
```

### Basic Usage

```python
from src.data.ocean_data import OceanDataGenerator, OceanHealthConfig
from src.models.ocean_models import train_ocean_health_models
from src.eval.ocean_evaluation import evaluate_ocean_models

# Generate sample ocean health data
config = OceanHealthConfig(n_samples=2000)
generator = OceanDataGenerator(config)
data = generator.generate_dataset()

# Prepare features for ML
from src.data.ocean_data import prepare_features
X, y = prepare_features(data)

# Train models
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

trainer = train_ocean_health_models(X_train, y_train, X_test, y_test)

# Evaluate models
evaluator = evaluate_ocean_models(trainer, X_test, y_test, data)

# Display results
leaderboard = evaluator.create_evaluation_report(evaluator.results)
print(leaderboard)
```

## Data Schema

The system works with ocean health monitoring data containing the following parameters:

### Core Parameters
- **Sea Surface Temperature** (°C): Critical for coral bleaching detection
- **Chlorophyll Concentration** (mg/m³): Indicator of algal blooms and productivity
- **pH Level**: Ocean acidification monitoring
- **Dissolved Oxygen** (mg/L): Essential for marine life survival
- **Salinity** (PSU): Water quality indicator

### Additional Parameters
- **Turbidity** (NTU): Water clarity measurement
- **Nitrate** (mg/L): Nutrient concentration
- **Phosphate** (mg/L): Nutrient concentration

### Spatial Features
- **Longitude/Latitude**: Geographic coordinates
- **Ocean Basin**: Pacific, Atlantic, Indian classification
- **Distance from Equator**: Simplified spatial feature

### Temporal Features
- **Date**: Temporal information
- **Month/Season**: Seasonal analysis capabilities

## Health Status Classification

The system classifies ocean regions into three health categories:

- **Healthy**: Normal ocean conditions with balanced parameters
- **Moderate Risk**: Some parameters outside normal ranges, monitoring recommended
- **Critical**: Severe conditions requiring immediate attention (coral bleaching, hypoxia, harmful algal blooms)

## Model Performance

The system includes multiple machine learning models:

- **Logistic Regression**: Baseline linear classifier
- **Random Forest**: Ensemble tree-based method
- **Gradient Boosting**: Advanced ensemble technique
- **XGBoost**: Optimized gradient boosting
- **Neural Network**: Deep learning PyTorch implementation
- **SVM**: Support vector machine classifier
- **K-Nearest Neighbors**: Instance-based learning

## Evaluation Metrics

### Standard ML Metrics
- Accuracy, Precision, Recall, F1-Score
- ROC AUC for multi-class classification
- Cross-validation with spatial stratification

### Domain-Specific Metrics
- **Critical Condition Detection**: Recall for severe health issues
- **False Alarm Rate**: Minimize false critical alerts
- **Healthy Preservation**: Precision for normal conditions
- **Temperature Performance**: Accuracy in high-temperature regions
- **Oxygen Performance**: Accuracy in low-oxygen conditions

### Spatial Metrics
- Performance by ocean basin
- Latitude/longitude band analysis
- Seasonal performance evaluation

## Project Structure

```
ocean-health-monitoring/
├── src/                    # Source code
│   ├── data/              # Data pipeline and generation
│   ├── models/            # Machine learning models
│   ├── eval/              # Evaluation metrics and analysis
│   └── viz/               # Visualization tools
├── demo/                  # Streamlit demo application
├── configs/               # Configuration files
├── data/                  # Data storage
│   ├── raw/              # Raw data files
│   ├── processed/        # Processed data
│   └── external/         # External data sources
├── assets/               # Generated visualizations and outputs
├── tests/                # Unit tests
├── notebooks/            # Jupyter notebooks for analysis
└── scripts/             # Utility scripts
```

## Configuration

The system uses YAML configuration files for customizable parameters:

- **Data Configuration**: Sample size, spatial extent, temporal range
- **Model Configuration**: Training parameters, cross-validation settings
- **Visualization Configuration**: Plot styles, color schemes
- **Geographic Configuration**: Coordinate systems, map settings

## Advanced Features

### Spatial Cross-Validation
Implements spatial block cross-validation to prevent data leakage in geographic models.

### Uncertainty Quantification
Provides confidence intervals and uncertainty estimates for predictions.

### Real-time Processing
Supports streaming data processing for continuous monitoring applications.

### Export Capabilities
- Interactive HTML maps
- High-resolution static plots
- Model performance reports
- Spatial analysis results

## Use Cases

- **Marine Conservation**: Monitor protected areas and ecosystem health
- **Early Warning Systems**: Detect coral bleaching and harmful algal blooms
- **Research Applications**: Analyze ocean health trends and patterns
- **Educational Tools**: Demonstrate ocean monitoring concepts
- **Policy Support**: Provide data-driven insights for marine management

## Technical Requirements

- Python 3.10+
- PyTorch 2.0+
- Scikit-learn 1.3+
- Geopandas for spatial analysis
- Streamlit for web interface
- Plotly/Folium for interactive visualizations

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This is a research and educational demonstration system. For operational ocean health monitoring, please:

- Validate with real-world data sources
- Consult with marine science experts
- Consider additional environmental factors
- Implement proper quality control measures
- Follow established monitoring protocols

The synthetic data used in this demonstration is for educational purposes only and should not be used for operational decision-making.

## Issues and Support

For questions, bug reports, or feature requests, please visit:
- **GitHub Issues**: https://github.com/kryptologyst/Ocean-Health-Monitoring-System/issues
- **Author**: kryptologyst - [GitHub](https://github.com/kryptologyst)

## Acknowledgments

- Marine science community for parameter thresholds and health criteria
- Open source ML and geospatial libraries
- Ocean monitoring research and datasets
- Environmental data science best practices
# Ocean-Health-Monitoring-System
