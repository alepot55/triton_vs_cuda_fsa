#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

VENV_NAME=".venv"
BASE_DIR="$(pwd)"

echo "=== SETUP TRITON VS CUDA FSA ENVIRONMENT ==="
echo "Creating virtual environment in $BASE_DIR/$VENV_NAME"

# Create virtual environment
python3 -m venv $VENV_NAME

# Activate virtual environment
source $VENV_NAME/bin/activate

# Update pip
pip install --upgrade pip

# Install required packages
echo "Installing required packages..."
pip install numpy pandas matplotlib seaborn torch triton

# Install additional analysis packages
pip install jupyter scikit-learn

# Check if CUDA toolkit is available
if command -v nvcc &> /dev/null; then
    echo "CUDA toolkit found: $(nvcc --version | head -n 1)"
else
    echo "WARNING: CUDA toolkit not found. Please install it to build and run CUDA components."
fi

# Create activation helper script
cat > activate_env.sh << EOF
#!/bin/bash
source $BASE_DIR/$VENV_NAME/bin/activate
echo "Virtual environment activated. Use 'deactivate' to exit."
EOF

chmod +x activate_env.sh

echo "=== SETUP COMPLETE ==="
echo "Virtual environment created in $VENV_NAME/"
echo "To activate the environment, run:"
echo "  source ./activate_env.sh"
echo ""
echo "Environment currently active. To deactivate, run:"
echo "  deactivate"
