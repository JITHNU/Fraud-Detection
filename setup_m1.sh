set -e


# Create & activate venv
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools


# Core scientific stack
pip install pandas numpy matplotlib scikit-learn scipy tqdm pyyaml joblib


# PyTorch CPU (works fine on M1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu


# PyTorch Geometric + companions (match torch version shown by: python -c "import torch; print(torch.__version__)")
# Example below assumes torch 2.3.x CPU. Adjust the version in the URL if needed.
TORCH_VER=$(python - << 'PY'
import torch
print(torch.__version__.split('+')[0])
PY
)


# Install PyG wheels compiled for CPU from official index
pip install --no-cache-dir torch-geometric
pip install --no-cache-dir torch-scatter==2.1.2 torch-sparse==0.6.18 torch-cluster==1.6.3 torch-spline-conv==1.2.2 \
-f https://data.pyg.org/whl/torch-${TORCH_VER}+cpu.html


# Streamlit for the app
pip install streamlit


# Optional: networkx for sanity checks
pip install networkx


echo "\n✅ Setup complete. Activate with: source .venv/bin/activate"