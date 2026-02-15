#!/bin/bash

echo "=========================================="
echo " RecoMart Pipeline Dependency Installer"
echo "=========================================="

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python 3 is not installed or not in your PATH."
    exit 1
fi

echo "[INFO] Python 3 found. Installing dependencies..."
echo ""

# Install dependencies
python -m pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo ""
    echo "[ERROR] Failed to install dependencies."
    exit 1
fi

echo ""
echo "[SUCCESS] All dependencies installed successfully!"
echo "You can now run the pipeline scripts."
echo ""
