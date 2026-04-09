#!/bin/bash

# Ocean Health Monitoring - Quick Setup Script
# This script sets up the environment and runs a basic test

echo "🌊 Ocean Health Monitoring - Quick Setup"
echo "========================================"

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Check if we're in the right directory
if [ ! -f "README.md" ]; then
    echo "❌ Error: Please run this script from the ocean-health-monitoring directory"
    exit 1
fi

echo "✅ Found project files"

# Create virtual environment (optional)
if [ "$1" = "--venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    echo "✅ Virtual environment created and activated"
fi

# Install basic dependencies
echo "📦 Installing basic dependencies..."
pip3 install numpy pandas scikit-learn matplotlib seaborn

# Test basic functionality
echo "🧪 Testing basic functionality..."
python3 simple_example.py

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 Setup completed successfully!"
    echo ""
    echo "Next steps:"
    echo "1. Install full dependencies: pip install -r requirements.txt"
    echo "2. Run the full example: python3 example.py"
    echo "3. Launch the demo: streamlit run demo/ocean_health_demo.py"
    echo ""
    echo "For more information, see README.md"
else
    echo "❌ Setup failed. Please check the error messages above."
    exit 1
fi
