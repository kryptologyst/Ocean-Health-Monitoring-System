"""
Ocean Health Monitoring Visualization

This module provides comprehensive visualization tools for ocean health monitoring,
including interactive maps, time series plots, and spatial analysis visualizations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from folium import plugins
import geopandas as gpd
from typing import Dict, List, Tuple, Optional, Any
import logging
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


class OceanHealthVisualizer:
    """Comprehensive visualization tools for ocean health monitoring."""
    
    def __init__(self, class_names: List[str] = None, 
                 class_colors: Dict[str, str] = None):
        self.class_names = class_names or ['Healthy', 'Moderate Risk', 'Critical']
        
        # Define colors for ocean health classes
        self.class_colors = class_colors or {
            'Healthy': '#2E8B57',      # Sea Green
            'Moderate Risk': '#FFD700', # Gold
            'Critical': '#DC143C'      # Crimson
        }
        
        # Set matplotlib style
        plt.style.use('default')
        sns.set_palette("husl")
    
    def create_ocean_health_map(self, data: pd.DataFrame, 
                              predictions: Optional[np.ndarray] = None,
                              model_name: str = "Model",
                              save_path: Optional[str] = None) -> folium.Map:
        """Create interactive Folium map showing ocean health status."""
        
        # Create base map centered on world oceans
        m = folium.Map(
            location=[0, 0],
            zoom_start=2,
            tiles='OpenStreetMap'
        )
        
        # Add different tile layers
        folium.TileLayer('CartoDB positron').add_to(m)
        folium.TileLayer('CartoDB dark_matter').add_to(m)
        
        # Prepare data for mapping
        map_data = data.copy()
        if predictions is not None:
            map_data['predicted_status'] = predictions
            map_data['predicted_label'] = map_data['predicted_status'].map({
                0: 'Healthy', 1: 'Moderate Risk', 2: 'Critical'
            })
            status_col = 'predicted_label'
        else:
            status_col = 'health_label'
        
        # Create color mapping
        color_map = {
            'Healthy': 'green',
            'Moderate Risk': 'orange', 
            'Critical': 'red'
        }
        
        # Add markers for each data point
        for idx, row in map_data.iterrows():
            color = color_map.get(row[status_col], 'gray')
            
            # Create popup text
            popup_text = f"""
            <b>Ocean Health Status</b><br>
            Location: {row['latitude']:.2f}°N, {row['longitude']:.2f}°E<br>
            Status: {row[status_col]}<br>
            Temperature: {row['sea_surface_temperature']:.1f}°C<br>
            Chlorophyll: {row['chlorophyll_concentration']:.2f} mg/m³<br>
            pH: {row['ph_level']:.2f}<br>
            Oxygen: {row['dissolved_oxygen']:.1f} mg/L<br>
            Salinity: {row['salinity']:.1f} PSU
            """
            
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=5,
                popup=folium.Popup(popup_text, max_width=300),
                color='black',
                weight=1,
                fillColor=color,
                fillOpacity=0.7
            ).add_to(m)
        
        # Add legend
        legend_html = '''
        <div style="position: fixed; 
                    bottom: 50px; left: 50px; width: 150px; height: 90px; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:14px; padding: 10px">
        <p><b>Ocean Health Status</b></p>
        <p><i class="fa fa-circle" style="color:green"></i> Healthy</p>
        <p><i class="fa fa-circle" style="color:orange"></i> Moderate Risk</p>
        <p><i class="fa fa-circle" style="color:red"></i> Critical</p>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))
        
        # Add layer control
        folium.LayerControl().add_to(m)
        
        if save_path:
            m.save(save_path)
            logger.info(f"Interactive map saved to {save_path}")
        
        return m
    
    def plot_ocean_parameters_distribution(self, data: pd.DataFrame,
                                         save_path: Optional[str] = None):
        """Plot distribution of ocean parameters by health status."""
        
        # Select key parameters
        params = [
            'sea_surface_temperature',
            'chlorophyll_concentration',
            'ph_level',
            'dissolved_oxygen',
            'salinity'
        ]
        
        param_labels = [
            'Sea Surface Temperature (°C)',
            'Chlorophyll Concentration (mg/m³)',
            'pH Level',
            'Dissolved Oxygen (mg/L)',
            'Salinity (PSU)'
        ]
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, (param, label) in enumerate(zip(params, param_labels)):
            for health_status in self.class_names:
                subset = data[data['health_label'] == health_status]
                axes[i].hist(subset[param], alpha=0.7, 
                           label=health_status, bins=30,
                           color=self.class_colors[health_status])
            
            axes[i].set_xlabel(label)
            axes[i].set_ylabel('Frequency')
            axes[i].set_title(f'Distribution of {label}')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        # Remove empty subplot
        axes[5].remove()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Parameter distributions saved to {save_path}")
        
        plt.show()
    
    def plot_correlation_heatmap(self, data: pd.DataFrame,
                               save_path: Optional[str] = None):
        """Plot correlation heatmap of ocean parameters."""
        
        # Select numeric columns
        numeric_cols = [
            'sea_surface_temperature',
            'chlorophyll_concentration',
            'ph_level',
            'dissolved_oxygen',
            'salinity',
            'turbidity',
            'nitrate',
            'phosphate'
        ]
        
        corr_data = data[numeric_cols].corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_data, annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.2f')
        plt.title('Correlation Matrix of Ocean Parameters')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Correlation heatmap saved to {save_path}")
        
        plt.show()
    
    def plot_spatial_distribution(self, data: pd.DataFrame,
                                save_path: Optional[str] = None):
        """Plot spatial distribution of ocean health status."""
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Scatter plot by health status
        for health_status in self.class_names:
            subset = data[data['health_label'] == health_status]
            axes[0].scatter(subset['longitude'], subset['latitude'],
                          c=self.class_colors[health_status], 
                          label=health_status, alpha=0.6, s=20)
        
        axes[0].set_xlabel('Longitude')
        axes[0].set_ylabel('Latitude')
        axes[0].set_title('Spatial Distribution of Ocean Health Status')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Histogram by latitude
        for health_status in self.class_names:
            subset = data[data['health_label'] == health_status]
            axes[1].hist(subset['latitude'], alpha=0.7, bins=20,
                        label=health_status, color=self.class_colors[health_status])
        
        axes[1].set_xlabel('Latitude')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Distribution by Latitude')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Spatial distribution plots saved to {save_path}")
        
        plt.show()
    
    def plot_temporal_analysis(self, data: pd.DataFrame,
                             save_path: Optional[str] = None):
        """Plot temporal analysis of ocean health parameters."""
        
        # Add temporal features
        data['month'] = data['date'].dt.month
        data['year'] = data['date'].dt.year
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Monthly distribution of health status
        monthly_health = data.groupby(['month', 'health_label']).size().unstack(fill_value=0)
        monthly_health.plot(kind='bar', ax=axes[0, 0], color=[self.class_colors[name] for name in monthly_health.columns])
        axes[0, 0].set_title('Monthly Distribution of Ocean Health Status')
        axes[0, 0].set_xlabel('Month')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].legend(title='Health Status')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Seasonal temperature trends
        seasonal_temp = data.groupby(['season', 'health_label'])['sea_surface_temperature'].mean().unstack()
        seasonal_temp.plot(kind='bar', ax=axes[0, 1], color=[self.class_colors[name] for name in seasonal_temp.columns])
        axes[0, 1].set_title('Seasonal Temperature by Health Status')
        axes[0, 1].set_xlabel('Season')
        axes[0, 1].set_ylabel('Average Temperature (°C)')
        axes[0, 1].legend(title='Health Status')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Monthly chlorophyll trends
        monthly_chl = data.groupby('month')['chlorophyll_concentration'].mean()
        axes[1, 0].plot(monthly_chl.index, monthly_chl.values, marker='o', linewidth=2)
        axes[1, 0].set_title('Monthly Chlorophyll Concentration Trends')
        axes[1, 0].set_xlabel('Month')
        axes[1, 0].set_ylabel('Chlorophyll (mg/m³)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Health status by ocean basin
        basin_health = data.groupby(['ocean_basin', 'health_label']).size().unstack(fill_value=0)
        basin_health.plot(kind='bar', ax=axes[1, 1], color=[self.class_colors[name] for name in basin_health.columns])
        axes[1, 1].set_title('Ocean Health Status by Basin')
        axes[1, 1].set_xlabel('Ocean Basin')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].legend(title='Health Status')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Temporal analysis plots saved to {save_path}")
        
        plt.show()
    
    def create_interactive_dashboard(self, data: pd.DataFrame,
                                  predictions: Optional[np.ndarray] = None,
                                  model_name: str = "Model") -> go.Figure:
        """Create interactive Plotly dashboard."""
        
        # Prepare data
        dashboard_data = data.copy()
        if predictions is not None:
            dashboard_data['predicted_status'] = predictions
            dashboard_data['predicted_label'] = dashboard_data['predicted_status'].map({
                0: 'Healthy', 1: 'Moderate Risk', 2: 'Critical'
            })
            status_col = 'predicted_label'
        else:
            status_col = 'health_label'
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Spatial Distribution', 'Parameter Correlations',
                          'Health Status Distribution', 'Temperature vs Oxygen'),
            specs=[[{"type": "scattergeo"}, {"type": "heatmap"}],
                   [{"type": "bar"}, {"type": "scatter"}]]
        )
        
        # Spatial distribution
        for health_status in self.class_names:
            subset = dashboard_data[dashboard_data[status_col] == health_status]
            fig.add_trace(
                go.Scattergeo(
                    lon=subset['longitude'],
                    lat=subset['latitude'],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=self.class_colors[health_status],
                        opacity=0.7
                    ),
                    name=health_status,
                    text=f"Status: {health_status}<br>Temp: {subset['sea_surface_temperature']:.1f}°C<br>Oxygen: {subset['dissolved_oxygen']:.1f} mg/L"
                ),
                row=1, col=1
            )
        
        # Parameter correlations
        numeric_cols = ['sea_surface_temperature', 'chlorophyll_concentration', 
                      'ph_level', 'dissolved_oxygen', 'salinity']
        corr_matrix = dashboard_data[numeric_cols].corr()
        
        fig.add_trace(
            go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu',
                zmid=0,
                text=corr_matrix.values,
                texttemplate="%{text:.2f}",
                textfont={"size": 10}
            ),
            row=1, col=2
        )
        
        # Health status distribution
        health_counts = dashboard_data[status_col].value_counts()
        fig.add_trace(
            go.Bar(
                x=health_counts.index,
                y=health_counts.values,
                marker_color=[self.class_colors[name] for name in health_counts.index],
                name="Health Status Count"
            ),
            row=2, col=1
        )
        
        # Temperature vs Oxygen scatter
        for health_status in self.class_names:
            subset = dashboard_data[dashboard_data[status_col] == health_status]
            fig.add_trace(
                go.Scatter(
                    x=subset['sea_surface_temperature'],
                    y=subset['dissolved_oxygen'],
                    mode='markers',
                    marker=dict(
                        color=self.class_colors[health_status],
                        size=8,
                        opacity=0.7
                    ),
                    name=health_status,
                    showlegend=False
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title=f"Ocean Health Monitoring Dashboard - {model_name}",
            height=800,
            showlegend=True
        )
        
        # Update geo subplot
        fig.update_geos(
            showframe=False,
            showcoastlines=True,
            projection_type="equirectangular"
        )
        
        return fig
    
    def save_all_visualizations(self, data: pd.DataFrame,
                              predictions: Optional[np.ndarray] = None,
                              model_name: str = "Model",
                              output_dir: str = "assets"):
        """Save all visualizations to output directory."""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving visualizations to {output_path}")
        
        # Create interactive map
        map_obj = self.create_ocean_health_map(data, predictions, model_name)
        map_obj.save(str(output_path / f"{model_name}_ocean_health_map.html"))
        
        # Create static plots
        self.plot_ocean_parameters_distribution(
            data, str(output_path / f"{model_name}_parameter_distributions.png")
        )
        
        self.plot_correlation_heatmap(
            data, str(output_path / f"{model_name}_correlation_heatmap.png")
        )
        
        self.plot_spatial_distribution(
            data, str(output_path / f"{model_name}_spatial_distribution.png")
        )
        
        self.plot_temporal_analysis(
            data, str(output_path / f"{model_name}_temporal_analysis.png")
        )
        
        # Create interactive dashboard
        dashboard = self.create_interactive_dashboard(data, predictions, model_name)
        dashboard.write_html(str(output_path / f"{model_name}_dashboard.html"))
        
        logger.info("All visualizations saved successfully")


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    from src.data.ocean_data import OceanDataGenerator, OceanHealthConfig
    
    # Generate sample data
    config = OceanHealthConfig(n_samples=1000)
    generator = OceanDataGenerator(config)
    data = generator.generate_dataset()
    
    # Create visualizer
    visualizer = OceanHealthVisualizer()
    
    # Create and save all visualizations
    visualizer.save_all_visualizations(data, model_name="Sample_Data")
    
    print("Visualization example completed!")
