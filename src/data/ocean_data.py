"""
Ocean Health Monitoring Data Pipeline

This module handles data generation, loading, and preprocessing for ocean health monitoring.
Supports both synthetic data generation and real-world data ingestion from various sources.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from typing import Tuple, Dict, List, Optional, Union
from pathlib import Path
import logging
from dataclasses import dataclass
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


@dataclass
class OceanHealthConfig:
    """Configuration for ocean health data generation."""
    n_samples: int = 1000
    n_regions: int = 50
    seed: int = 42
    spatial_extent: Tuple[float, float, float, float] = (-180, -90, 180, 90)  # lon_min, lat_min, lon_max, lat_max
    temporal_range: Tuple[str, str] = ("2020-01-01", "2023-12-31")


class OceanDataGenerator:
    """Generates synthetic ocean health monitoring data with spatial and temporal components."""
    
    def __init__(self, config: OceanHealthConfig):
        self.config = config
        self.rng = np.random.RandomState(config.seed)
        
    def generate_ocean_features(self) -> pd.DataFrame:
        """Generate synthetic ocean health features with realistic distributions."""
        n_samples = self.config.n_samples
        
        # Oceanographic parameters with realistic ranges
        sea_temp = self.rng.normal(26.0, 2.5, n_samples)  # °C (tropical to temperate)
        chlorophyll = self.rng.lognormal(0.3, 0.8, n_samples)  # mg/m³ (log-normal for biological data)
        ph_level = self.rng.normal(8.1, 0.15, n_samples)  # pH (ocean pH range)
        dissolved_oxygen = self.rng.normal(6.0, 1.2, n_samples)  # mg/L
        salinity = self.rng.normal(35.0, 1.5, n_samples)  # PSU (Practical Salinity Units)
        turbidity = self.rng.exponential(2.0, n_samples)  # NTU (Nephelometric Turbidity Units)
        nitrate = self.rng.exponential(0.5, n_samples)  # mg/L
        phosphate = self.rng.exponential(0.1, n_samples)  # mg/L
        
        # Spatial coordinates
        lon_min, lat_min, lon_max, lat_max = self.config.spatial_extent
        longitude = self.rng.uniform(lon_min, lon_max, n_samples)
        latitude = self.rng.uniform(lat_min, lat_max, n_samples)
        
        # Temporal features
        start_date, end_date = self.config.temporal_range
        dates = pd.date_range(start_date, end_date, periods=n_samples)
        
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
            'turbidity': turbidity,
            'nitrate': nitrate,
            'phosphate': phosphate
        })
        
        return data
    
    def generate_health_labels(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate ocean health status labels based on environmental thresholds."""
        # Define health thresholds based on marine science literature
        temp_critical = data['sea_surface_temperature'] > 28.5  # Coral bleaching threshold
        temp_moderate = data['sea_surface_temperature'] > 27.0
        
        oxygen_critical = data['dissolved_oxygen'] < 4.0  # Hypoxic conditions
        oxygen_moderate = data['dissolved_oxygen'] < 5.5
        
        chlorophyll_critical = data['chlorophyll_concentration'] > 3.0  # Harmful algal blooms
        chlorophyll_moderate = data['chlorophyll_concentration'] > 2.0
        
        ph_critical = (data['ph_level'] < 7.8) | (data['ph_level'] > 8.3)  # Ocean acidification
        ph_moderate = (data['ph_level'] < 8.0) | (data['ph_level'] > 8.2)
        
        # Combine conditions for health classification
        critical_conditions = (
            temp_critical | oxygen_critical | chlorophyll_critical | ph_critical
        )
        
        moderate_conditions = (
            temp_moderate | oxygen_moderate | chlorophyll_moderate | ph_moderate
        ) & ~critical_conditions
        
        # Assign health status: 0=Healthy, 1=Moderate Risk, 2=Critical
        health_status = np.where(critical_conditions, 2,
                                np.where(moderate_conditions, 1, 0))
        
        data['health_status'] = health_status
        data['health_label'] = data['health_status'].map({
            0: 'Healthy',
            1: 'Moderate Risk', 
            2: 'Critical'
        })
        
        return data
    
    def add_spatial_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add spatial features for enhanced modeling."""
        # Distance from coast (simplified - distance from equator)
        data['distance_from_equator'] = np.abs(data['latitude'])
        
        # Ocean basin classification (simplified)
        data['ocean_basin'] = pd.cut(
            data['longitude'], 
            bins=[-180, -60, 20, 180], 
            labels=['Pacific', 'Atlantic', 'Indian']
        )
        
        # Seasonal features
        data['month'] = data['date'].dt.month
        data['season'] = data['month'].map({
            12: 'Winter', 1: 'Winter', 2: 'Winter',
            3: 'Spring', 4: 'Spring', 5: 'Spring',
            6: 'Summer', 7: 'Summer', 8: 'Summer',
            9: 'Fall', 10: 'Fall', 11: 'Fall'
        })
        
        return data
    
    def generate_dataset(self) -> pd.DataFrame:
        """Generate complete ocean health monitoring dataset."""
        logger.info(f"Generating ocean health dataset with {self.config.n_samples} samples")
        
        # Generate base features
        data = self.generate_ocean_features()
        
        # Add health labels
        data = self.generate_health_labels(data)
        
        # Add spatial features
        data = self.add_spatial_features(data)
        
        logger.info(f"Dataset generated with shape: {data.shape}")
        logger.info(f"Health status distribution:\n{data['health_label'].value_counts()}")
        
        return data


def load_ocean_data(data_path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """Load ocean health data from file or generate synthetic data."""
    if data_path and Path(data_path).exists():
        logger.info(f"Loading ocean data from {data_path}")
        return pd.read_csv(data_path)
    else:
        logger.info("Generating synthetic ocean health data")
        config = OceanHealthConfig()
        generator = OceanDataGenerator(config)
        return generator.generate_dataset()


def prepare_features(data: pd.DataFrame, 
                    feature_cols: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare features and labels for machine learning."""
    if feature_cols is None:
        # Default feature columns (excluding spatial/temporal for basic ML)
        feature_cols = [
            'sea_surface_temperature',
            'chlorophyll_concentration', 
            'ph_level',
            'dissolved_oxygen',
            'salinity',
            'turbidity',
            'nitrate',
            'phosphate'
        ]
    
    X = data[feature_cols].values
    y = data['health_status'].values
    
    logger.info(f"Prepared features shape: {X.shape}")
    logger.info(f"Labels shape: {y.shape}")
    
    return X, y


def create_geodataframe(data: pd.DataFrame) -> gpd.GeoDataFrame:
    """Convert DataFrame to GeoDataFrame for spatial analysis."""
    gdf = gpd.GeoDataFrame(
        data,
        geometry=gpd.points_from_xy(data['longitude'], data['latitude']),
        crs='EPSG:4326'
    )
    return gdf


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    config = OceanHealthConfig(n_samples=2000)
    generator = OceanDataGenerator(config)
    data = generator.generate_dataset()
    
    print(f"Generated dataset shape: {data.shape}")
    print(f"Health status distribution:")
    print(data['health_label'].value_counts())
    
    # Save sample data
    data.to_csv('data/raw/ocean_health_sample.csv', index=False)
    print("Sample data saved to data/raw/ocean_health_sample.csv")
