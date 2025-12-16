# Replica-DATAWA

This project is a replication of the paper "DATA-WA: Demand-based Adaptive Task Assignment with Dynamic Worker Availability Windows". It implements a spatial crowdsourcing framework that predicts task demand and adaptively assigns tasks to workers.

## Project Status

### What has been done
1.  **Data Loading & Preprocessing**:
    *   Implemented `src/data_loader.py` to parse raw Yueche/DiDi datasets.
    *   Implemented `src/preprocess.py` to convert spatial-temporal data into grid-based multivariate time series for the prediction model.
2.  **Task Demand Prediction (DDGNN)**:
    *   Implemented the **Dynamic Dependency-based Graph Neural Network (DDGNN)** in `src/model.py`.
    *   Includes **Demand Dependency Learning** (learning dynamic adjacency matrix), **Dilated Causal Convolution** (capturing temporal trends), and **APPNP** (graph propagation).
    *   Implemented training loop in `src/train_prediction.py`.
3.  **Task Assignment Algorithms**:
    *   Implemented **Greedy Assignment** in `src/assignment.py`.
    *   Implemented **Dependency Graph Assignment (DTA)** in `src/assignment.py`.
        *   Constructs a Worker Dependency Graph based on reachable tasks.
        *   Decomposes the graph into connected components.
        *   Solves the assignment problem within each component using **Max Flow** (optimal for single-task assignment).
4.  **Simulation Framework**:
    *   Implemented `src/simulation.py` to simulate the arrival of workers and tasks over time and evaluate assignment algorithms.

### What is left to do
1.  **Task Value Function (TVF) & Reinforcement Learning**:
    *   The paper proposes `DATA-WA` which uses a Q-learning based TVF to evaluate the long-term value of assignments. This is currently not implemented.
2.  **Full Worker Dependency Separation (Tree Construction)**:
    *   The current DTA implementation uses connected components + Max Flow. The paper describes a more advanced **Recursive Tree Construction (RTC)** on maximal cliques to handle "Task Sequences" (more than one task per worker). Currently, we assume a sequence length of 1 (matching).
3.  **Integration of Prediction into Assignment**:
    *   The "DTA+TP" and "DATA-WA" methods use predicted tasks to guide current assignments. The current simulation only uses actual revealed tasks.
4.  **Parameter Tuning**:
    *   Hyperparameters for the model and simulation (grid size, time intervals) need to be aligned strictly with the paper's experimental setup.

## Implementation Details vs. Paper

### Strictly Following Paper
*   **DDGNN Architecture**: The model architecture (Demand Dependency Learning -> Dilated Causal Conv -> APPNP) follows the paper's description in Section III.
*   **Data Handling**: We use the Yueche/DiDi datasets and follow the grid-based partition strategy.
*   **Framework**: The overall flow of "Adaptive Task Assignment" (handling events as they arrive) is followed.

### Inferred / Simplified
*   **DTA Implementation**: The paper describes `DTA` using "Worker Dependency Separation" with a clique-tree structure to optimize *sequences* of tasks. To get a working baseline quickly, we implemented `DTA` using **Max Flow** on connected components. This is mathematically optimal for the "Matching" problem (1 task per worker) but simplifies the "Sequence" aspect.
*   **Grid Size**: We estimated grid sizes based on data bounds (approx 0.005 degrees), whereas the paper might use specific grid dimensions (e.g., 1km x 1km).
*   **Time Interval**: We used a default $\Delta T$ of 5s or similar for prediction, as per the paper's default, but this can be adjusted.

## Folder Structure

*   `data/`: Contains raw and processed data.
    *   `raw/`: Original datasets (Yueche/DiDi).
    *   `processed/`: Pickled data files for quick loading.
*   `latex/`: The original LaTeX source code of the paper.
*   `src/`: Source code for the implementation.
    *   `data_loader.py`: Loads raw text data.
    *   `preprocess.py`: Generates grid data and time series.
    *   `model.py`: PyTorch implementation of DDGNN.
    *   `train_prediction.py`: Training script for DDGNN.
    *   `assignment.py`: Implementation of Greedy and DTA algorithms.
    *   `simulation.py`: Main simulation loop.
*   `models/`: Directory to save trained model weights.

## How to Run

1.  **Prerequisites**:
    *   Python 3.x
    *   PyTorch, NumPy

2.  **Preprocessing**:
    ```bash
    python src/preprocess.py
    ```
    This reads `data/raw`, filters data, creates grid features, and saves to `data/processed/yueche_data.pkl`.

3.  **Train Prediction Model**:
    ```bash
    python src/train_prediction.py
    ```
    Trains the DDGNN model and saves it to `models/`.

4.  **Run Simulation**:
    ```bash
    python src/simulation.py
    ```
    Runs the adaptive task assignment simulation. You can modify the `method` in `src/simulation.py` to switch between `'greedy'` and `'dta'`.

# Discoveries

1.  **Grid Size Impact**:
    * The grid size has a significant impact on the model's ability to capture spatial dependencies. A smaller grid size (e.g., 1km x 1km) allows the model to learn more fine-grained patterns but requires more data. 
2. Traing Details:
   * Focal Loss + WeightedSampler may help balance the class imbalance in the training data. 
3. 