# Server Deployment Guide

This guide explains how to deploy the Replica-DATAWA project on a remote server (e.g., Linux GPU server).

## 1. Prerequisites

Ensure your server has:
- **Python 3.8+** (Anaconda/Miniconda recommended)
- **CUDA** (if using GPU)
- **Git**

## 2. Setup Environment

```bash
# Clone the repository
git clone <your-repo-url>
cd Replica-DATAWA

# Create and activate conda environment
conda create -n replica python=3.9 -y
conda activate replica

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  # Adjust CUDA version
pip install numpy pandas matplotlib seaborn scikit-learn tqdm tensorboard
pip install streamlit folium streamlit-folium
```

## 3. Running Prediction Experiments (Training)

You can run the training script in the background using `nohup`.

**Option 1: Quick Test (CPU/GPU)**
```bash
export QUICK=1
export GRID_MODE=1km  # or 0.02deg
nohup python src/run_prediction_experiments.py > experiment.log 2>&1 &
```

**Option 2: Full Experiment**
```bash
export QUICK=0
export GRID_MODE=1km
nohup python src/run_prediction_experiments.py > experiment_full.log 2>&1 &
```

Check progress:
```bash
tail -f experiment.log
```

## 4. Running the Visualization Dashboard

To run the Streamlit app on a server and access it locally:

```bash
# Run on server (port 8501 by default)
nohup streamlit run app/dashboard.py --server.port 8501 --server.headless true > dashboard.log 2>&1 &
```

### Accessing the Dashboard

If your server has a public IP and the port is open:
- Go to `http://<server-ip>:8501`

If your server is behind a firewall (e.g., SSH only):
- Use SSH Tunneling on your **local machine**:
  ```bash
  ssh -L 8501:localhost:8501 user@your-server-ip
  ```
- Then open `http://localhost:8501` in your browser.

## 5. Project Structure

- `src/`: Core logic (Preprocessing, Models, Training Loop).
- `app/`: Visualization Dashboard (Streamlit).
- `data/`: Raw and processed data.
- `results/`: Experiment outputs and logs.
