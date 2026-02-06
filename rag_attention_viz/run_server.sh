#!/bin/bash

# RAG + Attention Visualization Server Startup Script

echo "======================================"
echo "  RAG + Attention Visualization"
echo "  Starting Server..."
echo "======================================"
echo ""

# Check if we're in the correct directory
if [ ! -d "backend" ]; then
    echo "Error: Please run this script from the rag_attention_viz directory"
    exit 1
fi

# Check if Python is available
if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    echo "Error: Python is not installed or not in PATH"
    exit 1
fi

# Use python3 if available, otherwise python
PYTHON_CMD="python3"
if ! command -v python3 &> /dev/null; then
    PYTHON_CMD="python"
fi

echo "Using Python: $PYTHON_CMD"
echo ""

# Check if requirements are installed
echo "Checking dependencies..."
$PYTHON_CMD -c "import fastapi, transformers, torch, faiss" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Warning: Some dependencies may not be installed."
    echo "Run: pip install -r requirements.txt"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Change to backend directory
cd backend

echo "Starting FastAPI server..."
echo ""
echo "Once started, open your browser and navigate to:"
echo "  http://localhost:8000"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""
echo "======================================"
echo ""

# Run the server
$PYTHON_CMD app.py

echo ""
echo "Server stopped."
