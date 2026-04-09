"""
Ocean Health Monitoring Demo Application

Interactive Streamlit application for ocean health monitoring with real-time
visualization, model comparison, and spatial analysis capabilities.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from folium import plugins
import geopandas as gpd
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
import sys
import warnings

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data.ocean_data import OceanDataGenerator, OceanHealthConfig, prepare_features
from src.models.ocean_models import train_ocean_health_models, ModelConfig
from src.eval.ocean_evaluation import evaluate_ocean_models, OceanHealthEvaluator
from src.viz.ocean_visualization import OceanHealthVisualizer

warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Ocean Health Monitoring",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stAlert {
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

def load_sample_data():
    """Load or generate sample ocean health data."""
    config = OceanHealthConfig(n_samples=2000)
    generator = OceanDataGenerator(config)
    return generator.generate_dataset()

def create_model_comparison_chart(results: Dict):
    """Create model comparison chart."""
    model_names = list(results.keys())
    accuracies = [results[name]['test_score'] for name in model_names]
    
    fig = go.Figure(data=[
        go.Bar(x=model_names, y=accuracies, 
               marker_color=['#2E8B57', '#FFD700', '#DC143C', '#4169E1', '#FF6347'])
    ])
    
    fig.update_layout(
        title="Model Performance Comparison",
        xaxis_title="Model",
        yaxis_title="Accuracy",
        yaxis=dict(range=[0, 1])
    )
    
    return fig

def create_spatial_map(data: pd.DataFrame, predictions: np.ndarray = None):
    """Create interactive spatial map."""
    if predictions is not None:
        data = data.copy()
        data['predicted_status'] = predictions
        data['predicted_label'] = data['predicted_status'].map({
            0: 'Healthy', 1: 'Moderate Risk', 2: 'Critical'
        })
        status_col = 'predicted_label'
        title = "Predicted Ocean Health Status"
    else:
        status_col = 'health_label'
        title = "Actual Ocean Health Status"
    
    # Create color mapping
    color_map = {
        'Healthy': 'green',
        'Moderate Risk': 'orange',
        'Critical': 'red'
    }
    
    # Create map
    m = folium.Map(
        location=[0, 0],
        zoom_start=2,
        tiles='OpenStreetMap'
    )
    
    # Add markers
    for idx, row in data.iterrows():
        color = color_map.get(row[status_col], 'gray')
        
        popup_text = f"""
        <b>{title}</b><br>
        Location: {row['latitude']:.2f}°N, {row['longitude']:.2f}°E<br>
        Status: {row[status_col]}<br>
        Temperature: {row['sea_surface_temperature']:.1f}°C<br>
        Chlorophyll: {row['chlorophyll_concentration']:.2f} mg/m³<br>
        pH: {row['ph_level']:.2f}<br>
        Oxygen: {row['dissolved_oxygen']:.1f} mg/L
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
    
    return m

def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">🌊 Ocean Health Monitoring System</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    This interactive dashboard provides comprehensive ocean health monitoring capabilities,
    including real-time analysis, model comparison, and spatial visualization of marine ecosystems.
    """)
    
    # Sidebar
    st.sidebar.title("Configuration")
    
    # Data parameters
    st.sidebar.subheader("Data Parameters")
    n_samples = st.sidebar.slider("Number of Samples", 500, 5000, 2000)
    random_seed = st.sidebar.number_input("Random Seed", value=42, min_value=1, max_value=1000)
    
    # Model parameters
    st.sidebar.subheader("Model Parameters")
    test_size = st.sidebar.slider("Test Size", 0.1, 0.4, 0.2)
    cv_folds = st.sidebar.slider("Cross-Validation Folds", 3, 10, 5)
    
    # Load data
    with st.spinner("Loading ocean health data..."):
        config = OceanHealthConfig(n_samples=n_samples, seed=random_seed)
        generator = OceanDataGenerator(config)
        data = load_sample_data()
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview", "🗺️ Spatial Analysis", "🤖 Model Comparison", 
        "📈 Detailed Analysis", "ℹ️ About"
    ])
    
    with tab1:
        st.header("Ocean Health Overview")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Samples", f"{len(data):,}")
        
        with col2:
            healthy_count = len(data[data['health_label'] == 'Healthy'])
            st.metric("Healthy Regions", f"{healthy_count:,}")
        
        with col3:
            moderate_count = len(data[data['health_label'] == 'Moderate Risk'])
            st.metric("Moderate Risk", f"{moderate_count:,}")
        
        with col4:
            critical_count = len(data[data['health_label'] == 'Critical'])
            st.metric("Critical Regions", f"{critical_count:,}")
        
        # Health status distribution
        st.subheader("Ocean Health Status Distribution")
        
        col1, col2 = st.columns(2)
        
        with col1:
            health_counts = data['health_label'].value_counts()
            fig_pie = px.pie(
                values=health_counts.values,
                names=health_counts.index,
                title="Health Status Distribution",
                color_discrete_map={
                    'Healthy': '#2E8B57',
                    'Moderate Risk': '#FFD700',
                    'Critical': '#DC143C'
                }
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            fig_bar = px.bar(
                x=health_counts.index,
                y=health_counts.values,
                title="Health Status Counts",
                color=health_counts.index,
                color_discrete_map={
                    'Healthy': '#2E8B57',
                    'Moderate Risk': '#FFD700',
                    'Critical': '#DC143C'
                }
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        
        # Parameter distributions
        st.subheader("Ocean Parameter Distributions")
        
        param_cols = ['sea_surface_temperature', 'chlorophyll_concentration', 
                    'ph_level', 'dissolved_oxygen', 'salinity']
        
        selected_param = st.selectbox("Select Parameter", param_cols)
        
        fig_dist = px.histogram(
            data, x=selected_param, color='health_label',
            title=f"Distribution of {selected_param.replace('_', ' ').title()}",
            color_discrete_map={
                'Healthy': '#2E8B57',
                'Moderate Risk': '#FFD700',
                'Critical': '#DC143C'
            }
        )
        st.plotly_chart(fig_dist, use_container_width=True)
    
    with tab2:
        st.header("Spatial Analysis")
        
        # Map visualization
        st.subheader("Ocean Health Map")
        
        map_type = st.radio("Map Type", ["Actual Status", "Model Predictions"], horizontal=True)
        
        if map_type == "Actual Status":
            m = create_spatial_map(data)
        else:
            # Train a quick model for predictions
            with st.spinner("Training model for predictions..."):
                X, y = prepare_features(data)
                from sklearn.model_selection import train_test_split
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_seed, stratify=y
                )
                
                model_config = ModelConfig(random_state=random_seed, test_size=test_size)
                trainer = train_ocean_health_models(X_train, y_train, X_test, y_test, model_config)
                
                # Get predictions for all data
                best_model_name = max(trainer.results.keys(), 
                                    key=lambda x: trainer.results[x]['test_score'])
                best_model = trainer.results[best_model_name]['model']
                
                if best_model_name == 'neural_network':
                    import torch
                    X_tensor = torch.FloatTensor(X)
                    best_model.eval()
                    with torch.no_grad():
                        outputs = best_model(X_tensor)
                        predictions = torch.argmax(outputs, dim=1).numpy()
                else:
                    predictions = best_model.predict(X)
            
            m = create_spatial_map(data, predictions)
        
        # Display map
        st.components.v1.html(m._repr_html_(), height=600)
        
        # Spatial statistics
        st.subheader("Spatial Statistics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # By ocean basin
            basin_stats = data.groupby('ocean_basin')['health_label'].value_counts().unstack(fill_value=0)
            fig_basin = px.bar(
                basin_stats, 
                title="Health Status by Ocean Basin",
                color_discrete_map={
                    'Healthy': '#2E8B57',
                    'Moderate Risk': '#FFD700',
                    'Critical': '#DC143C'
                }
            )
            st.plotly_chart(fig_basin, use_container_width=True)
        
        with col2:
            # By latitude bands
            data['lat_band'] = pd.cut(data['latitude'], bins=5, labels=['S5', 'S4', 'S3', 'S2', 'S1'])
            lat_stats = data.groupby('lat_band')['health_label'].value_counts().unstack(fill_value=0)
            fig_lat = px.bar(
                lat_stats,
                title="Health Status by Latitude Band",
                color_discrete_map={
                    'Healthy': '#2E8B57',
                    'Moderate Risk': '#FFD700',
                    'Critical': '#DC143C'
                }
            )
            st.plotly_chart(fig_lat, use_container_width=True)
    
    with tab3:
        st.header("Model Comparison")
        
        # Train models
        with st.spinner("Training multiple models for comparison..."):
            X, y = prepare_features(data)
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_seed, stratify=y
            )
            
            model_config = ModelConfig(random_state=random_seed, test_size=test_size, cv_folds=cv_folds)
            trainer = train_ocean_health_models(X_train, y_train, X_test, y_test, model_config)
            
            # Evaluate models
            evaluator = evaluate_ocean_models(trainer, X_test, y_test, data)
        
        # Model comparison chart
        st.subheader("Model Performance Comparison")
        
        comparison_data = []
        for model_name, results in trainer.results.items():
            comparison_data.append({
                'Model': model_name.replace('_', ' ').title(),
                'Test Accuracy': results['test_score'],
                'CV Mean': results['cv_mean'],
                'CV Std': results['cv_std']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('Test Accuracy', ascending=False)
        
        fig_comparison = px.bar(
            comparison_df, x='Model', y='Test Accuracy',
            title="Model Accuracy Comparison",
            color='Test Accuracy',
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig_comparison, use_container_width=True)
        
        # Detailed results table
        st.subheader("Detailed Model Results")
        st.dataframe(comparison_df, use_container_width=True)
        
        # Best model predictions
        best_model_name = comparison_df.iloc[0]['Model'].lower().replace(' ', '_')
        if best_model_name in trainer.results:
            st.subheader(f"Best Model: {best_model_name.title()}")
            
            best_results = trainer.results[best_model_name]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Test Accuracy", f"{best_results['test_score']:.4f}")
                st.metric("CV Mean", f"{best_results['cv_mean']:.4f}")
                st.metric("CV Std", f"{best_results['cv_std']:.4f}")
            
            with col2:
                st.text("Classification Report:")
                st.text(best_results['classification_report'])
    
    with tab4:
        st.header("Detailed Analysis")
        
        # Correlation analysis
        st.subheader("Parameter Correlations")
        
        numeric_cols = ['sea_surface_temperature', 'chlorophyll_concentration', 
                      'ph_level', 'dissolved_oxygen', 'salinity', 'turbidity']
        
        corr_matrix = data[numeric_cols].corr()
        
        fig_corr = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            title="Parameter Correlation Matrix",
            color_continuous_scale='RdBu'
        )
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # Temporal analysis
        st.subheader("Temporal Analysis")
        
        data['month'] = data['date'].dt.month
        monthly_stats = data.groupby(['month', 'health_label']).size().unstack(fill_value=0)
        
        fig_temporal = px.line(
            monthly_stats,
            title="Monthly Health Status Trends",
            color_discrete_map={
                'Healthy': '#2E8B57',
                'Moderate Risk': '#FFD700',
                'Critical': '#DC143C'
            }
        )
        st.plotly_chart(fig_temporal, use_container_width=True)
        
        # Parameter relationships
        st.subheader("Parameter Relationships")
        
        param1 = st.selectbox("Parameter 1", numeric_cols, key="param1")
        param2 = st.selectbox("Parameter 2", numeric_cols, key="param2")
        
        if param1 != param2:
            fig_scatter = px.scatter(
                data, x=param1, y=param2, color='health_label',
                title=f"{param1.replace('_', ' ').title()} vs {param2.replace('_', ' ').title()}",
                color_discrete_map={
                    'Healthy': '#2E8B57',
                    'Moderate Risk': '#FFD700',
                    'Critical': '#DC143C'
                }
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
    
    with tab5:
        st.header("About Ocean Health Monitoring")
        
        st.markdown("""
        ### Overview
        This Ocean Health Monitoring System provides comprehensive analysis and visualization
        of marine ecosystem health using advanced machine learning techniques and spatial analysis.
        
        ### Key Features
        - **Real-time Analysis**: Process ocean health data in real-time
        - **Spatial Visualization**: Interactive maps showing health status across regions
        - **Model Comparison**: Compare multiple ML models for optimal performance
        - **Comprehensive Metrics**: Domain-specific evaluation metrics for ocean health
        
        ### Ocean Health Parameters
        - **Sea Surface Temperature**: Critical for coral bleaching detection
        - **Chlorophyll Concentration**: Indicator of algal blooms and productivity
        - **pH Level**: Ocean acidification monitoring
        - **Dissolved Oxygen**: Essential for marine life survival
        - **Salinity**: Water quality indicator
        
        ### Health Status Classification
        - **Healthy**: Normal ocean conditions with balanced parameters
        - **Moderate Risk**: Some parameters outside normal ranges
        - **Critical**: Severe conditions requiring immediate attention
        
        ### Technical Implementation
        - **Data Pipeline**: Synthetic ocean data generation with realistic distributions
        - **Machine Learning**: Multiple algorithms including neural networks and ensemble methods
        - **Spatial Analysis**: Geographic visualization and regional analysis
        - **Interactive Dashboard**: Streamlit-based web application
        
        ### Use Cases
        - Marine conservation monitoring
        - Early warning systems for coral bleaching
        - Harmful algal bloom detection
        - Ocean acidification tracking
        - Marine ecosystem health assessment
        
        ### Disclaimer
        This is a research and educational demonstration. For operational use,
        please consult with marine science experts and validate with real-world data.
        """)
        
        st.markdown("""
        ### Author
        **kryptologyst** - [GitHub](https://github.com/kryptologyst)
        
        ### Data Sources
        This demonstration uses synthetic ocean health data generated with realistic
        parameter distributions based on marine science literature.
        """)

if __name__ == "__main__":
    main()
