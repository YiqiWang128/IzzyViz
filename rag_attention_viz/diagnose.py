#!/usr/bin/env python3
"""
Diagnostic script to check what's missing for RAG + Attention Visualization
"""
import sys
import subprocess

print("="*60)
print("RAG + Attention Visualization - Diagnostic Tool")
print("="*60)
print()

print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")
print()

packages_to_check = [
    'pandas',
    'matplotlib',
    'numpy',
    'torch',
    'transformers',
    'sentence_transformers',
    'faiss',
    'fastapi',
    'uvicorn'
]

print("Checking packages:")
print("-"*60)
missing = []
installed = []

for pkg in packages_to_check:
    try:
        __import__(pkg.replace('-', '_'))
        print(f"✓ {pkg:25s} installed")
        installed.append(pkg)
    except ImportError as e:
        print(f"✗ {pkg:25s} MISSING")
        missing.append(pkg)

print("-"*60)
print()

if missing:
    print(f"❌ Missing {len(missing)} package(s): {', '.join(missing)}")
    print()
    print("To install missing packages, run:")
    print()
    print(f"  {sys.executable} -m pip install {' '.join(missing)}")
    print()
    print("Or install all requirements:")
    print()
    print(f"  {sys.executable} -m pip install -r requirements.txt")
    print()
    sys.exit(1)
else:
    print(f"✓ All {len(installed)} required packages are installed!")
    print()

    # Test IzzyViz import
    print("Testing IzzyViz import:")
    print("-"*60)
    try:
        from izzyviz import compare_two_attentions_with_circles, visualize_attention_self_attention
        print("✓ IzzyViz imported successfully")
        print("✓ compare_two_attentions_with_circles available")
        print("✓ visualize_attention_self_attention available")
    except ImportError as e:
        print(f"✗ IzzyViz import failed: {e}")
        print()
        print("IzzyViz needs to be installed. Run:")
        print()
        print("  cd /home/user/Wing-IzzyViz")
        print(f"  {sys.executable} -m pip install -e .")
        print()
        sys.exit(1)

    print("-"*60)
    print()
    print("="*60)
    print("✓ System is ready to run!")
    print("="*60)
    print()
    print("Start the server with:")
    print("  cd backend && python app.py")
    print()
