#!/bin/bash

echo "============================================"
echo "  RAG + Attention Visualization"
echo "  Quick Fix for Missing Dependencies"
echo "============================================"
echo ""

# Check current directory
if [ ! -f "requirements.txt" ]; then
    echo "Error: Please run this script from the rag_attention_viz directory"
    exit 1
fi

echo "Step 1: Installing IzzyViz dependencies..."
cd ..
pip install -e . || {
    echo "Failed to install IzzyViz"
    exit 1
}
cd rag_attention_viz

echo ""
echo "Step 2: Installing RAG project dependencies..."
pip install -r requirements.txt || {
    echo "Failed to install requirements"
    exit 1
}

echo ""
echo "Step 3: Verifying installation..."
python3 -c "
import sys
try:
    import pandas
    print('✓ pandas installed')
except ImportError:
    print('✗ pandas NOT installed')
    sys.exit(1)

try:
    import matplotlib
    print('✓ matplotlib installed')
except ImportError:
    print('✗ matplotlib NOT installed')
    sys.exit(1)

try:
    from izzyviz import compare_two_attentions_with_circles
    print('✓ IzzyViz imported successfully')
except ImportError as e:
    print(f'✗ IzzyViz import failed: {e}')
    sys.exit(1)

try:
    import torch
    print('✓ torch installed')
except ImportError:
    print('✗ torch NOT installed')
    sys.exit(1)

try:
    import transformers
    print('✓ transformers installed')
except ImportError:
    print('✗ transformers NOT installed')
    sys.exit(1)

try:
    import faiss
    print('✓ faiss installed')
except ImportError:
    print('✗ faiss NOT installed')
    sys.exit(1)

try:
    import fastapi
    print('✓ fastapi installed')
except ImportError:
    print('✗ fastapi NOT installed')
    sys.exit(1)

print('')
print('All dependencies verified successfully!')
" || {
    echo ""
    echo "Some dependencies are still missing."
    echo "Please run manually:"
    echo "  cd .. && pip install -e . && cd rag_attention_viz"
    echo "  pip install -r requirements.txt"
    exit 1
}

echo ""
echo "============================================"
echo "  ✓ All dependencies installed!"
echo "============================================"
echo ""
echo "You can now start the server:"
echo "  cd backend && python app.py"
echo ""
