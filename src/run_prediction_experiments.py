import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import random
import time
import json
import os
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import average_precision_score
from preprocess import Preprocessor
from model import DDGNN
from baselines import LSTMModel, GraphWaveNet

# logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("experiment.log"),
        logging.StreamHandler()
    ]
)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
try:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
except Exception:
    pass

# Hyperparameters
CONFIG = {
    'epochs': 120,
    'patience': 12,
    'batch_size': 64,
    'learning_rate': 0.001,
    'seq_len': 60,
    'train_split': 0.8, # Increase train split to ~19.2h (for 24h data)
    'val_split': 0.05,  # ~1.2h
    'test_split': 0.15, # ~3.6h
    'clip_grad': 5.0, # Gradient clipping
}

if os.getenv('QUICK') == '1':
    CONFIG['epochs'] = 10
    CONFIG['patience'] = 5

def create_dataset(data, seq_len):
    # data: (Total_Time, Num_Nodes, k)
    X, Y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        Y.append(data[i+seq_len])
    return torch.FloatTensor(np.array(X)), torch.FloatTensor(np.array(Y))

class FocalBCEWithLogitsLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, pos_weight=None, node_weights=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight
        self.node_weights = node_weights

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=self.pos_weight, reduction='none')
        probs = torch.sigmoid(logits)
        pt = probs * targets + (1 - probs) * (1 - targets)
        loss = self.alpha * (1 - pt) ** self.gamma * bce
        if self.node_weights is not None:
            w = self.node_weights.view(1, -1, 1).to(logits.device)
            loss = loss * w
        return loss.mean()
def train_model(model, train_loader, val_loader, criterion, optimizer, device, writer, tag_prefix, model_name_only, config):
    model.train()
    start_time = time.time()
    
    global_step = 0
    steps_per_epoch = len(train_loader)
    
    best_val_ap = float('-inf')
    best_model_state = None
    patience_counter = 0
    
    # Outer progress bar for epochs
    epoch_pbar = tqdm(range(config['epochs']), desc=f"Training {tag_prefix}", position=0, leave=True)
    verbose_epochs = os.getenv('VERBOSE_EPOCHS') == '1'
    
    for epoch in epoch_pbar:
        # Training
        model.train()
        total_loss = 0
        num_batches = 0
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            
            if isinstance(model, DDGNN):
                out, _ = model(x)
            elif isinstance(model, GraphWaveNet):
                out, _ = model(x)
            else:
                out, _ = model(x)
            
            loss = criterion(out, y)
            loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['clip_grad'])
            
            optimizer.step()
            
            current_loss = loss.item()
            total_loss += current_loss
            num_batches += 1
            
            # Log step loss
            global_step_val = epoch * steps_per_epoch + num_batches
            writer.add_scalar(f'{tag_prefix}/Loss/Step/{model_name_only}', current_loss, global_step_val)
            
        avg_loss = total_loss / num_batches
        writer.add_scalar(f'{tag_prefix}/Loss/Epoch/Train/{model_name_only}', avg_loss, epoch)
        
        # Validation
        val_loss = validate_model(model, val_loader, criterion, device)
        writer.add_scalar(f'{tag_prefix}/Loss/Epoch/Val/{model_name_only}', val_loss, epoch)
        val_ap, _, preds_val, targets_val, _ = evaluate_model(model, val_loader, device)
        if os.getenv('STRICT_AP') == '1':
            preds_np = preds_val
            targets_np = targets_val
            num_nodes_eval = preds_np.shape[1]
            node_aps = []
            for node_idx in range(num_nodes_eval):
                p = preds_np[:, node_idx, :].reshape(-1)
                t = targets_np[:, node_idx, :].reshape(-1)
                try:
                    node_ap = average_precision_score(t, p)
                except Exception:
                    node_ap = 0.0
                node_aps.append(node_ap)
            val_ap = float(np.mean(node_aps))
        writer.add_scalar(f'{tag_prefix}/Metrics/ValAP/{model_name_only}', val_ap, epoch)
        
        # Update epoch pbar
        epoch_pbar.set_postfix({'TrainLoss': f"{avg_loss:.4f}", 'ValLoss': f"{val_loss:.4f}", 'ValAP': f"{val_ap:.4f}"})
        if verbose_epochs:
            logging.info(f"[{tag_prefix}] Epoch {epoch+1}/{config['epochs']} - Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}, Val AP: {val_ap:.4f}")
        
        # Early Stopping
        if val_ap > best_val_ap:
            best_val_ap = val_ap
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config['patience']:
                logging.info(f"Early stopping at epoch {epoch+1}")
                break
        
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        logging.info(f"Loaded best model with Val AP: {best_val_ap:.4f}")

    end_time = time.time()
    return end_time - start_time

def validate_model(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    num_batches = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            if isinstance(model, DDGNN):
                out, _ = model(x)
            elif isinstance(model, GraphWaveNet):
                out, _ = model(x)
            else:
                out, _ = model(x)
            
            loss = criterion(out, y)
            total_loss += loss.item()
            num_batches += 1
    return total_loss / num_batches if num_batches > 0 else 0

def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_targets = []
    all_inputs = [] # Capture inputs for visualization
    start_time = time.time()
    with torch.no_grad():
        for x, y in test_loader:
            x_cpu = x.cpu().numpy()
            x, y = x.to(device), y.to(device)
            if isinstance(model, DDGNN):
                out, _ = model(x)
            elif isinstance(model, GraphWaveNet):
                out, _ = model(x)
            else:
                out, _ = model(x)
            
            # Sigmoid for probability
            probs = torch.sigmoid(out)
            all_preds.append(probs.cpu().numpy())
            all_targets.append(y.cpu().numpy())
            all_inputs.append(x_cpu)
            
    end_time = time.time()
    
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    all_inputs = np.concatenate(all_inputs)
    
    # Flatten for AP calculation
    ap = average_precision_score(all_targets.flatten(), all_preds.flatten())
    return ap, end_time - start_time, all_preds, all_targets, all_inputs

def visualize_cases(model_name, inputs, preds, targets, grid_shape, dataset_name, delta_t):
    # inputs: (Num_Samples, Seq_Len, Num_Nodes, Features)
    # preds: (Num_Samples, Num_Nodes, Output_Dim) - Probabilities
    # targets: (Num_Samples, Num_Nodes, Output_Dim) - Binary
    
    # We want to find "Bad Cases"
    # Metrics:
    # 1. False Positives (Sum of Pred where Target=0)
    # 2. False Negatives (Sum of (1-Pred) where Target=1)
    
    # Aggregating over nodes and output_dim
    # preds > 0.5 is threshold for binary decision usually, but let's use raw prob error
    
    # Error Map: (Pred - Target)^2
    # But we want specifically FP and FN
    
    # Flatten to (Num_Samples, -1)
    preds_flat = preds.reshape(preds.shape[0], -1)
    targets_flat = targets.reshape(targets.shape[0], -1)
    
    # Calculate FP score per sample: Sum(Pred * (1-Target))
    fp_scores = (preds_flat * (1 - targets_flat)).sum(axis=1)
    
    # Calculate FN score per sample: Sum((1-Pred) * Target)
    fn_scores = ((1 - preds_flat) * targets_flat).sum(axis=1)
    
    # Get Top 5 indices
    top_fp_indices = np.argsort(fp_scores)[-5:][::-1]
    top_fn_indices = np.argsort(fn_scores)[-5:][::-1]
    
    lat_steps, lng_steps = grid_shape
    
    save_dir = f'analysis_results/{dataset_name}_dT{delta_t}/{model_name}'
    os.makedirs(save_dir, exist_ok=True)
    
    def plot_sample(idx, case_type):
        # Input: Sum of demand over last 6 steps (30 mins if dt=5)
        # inputs[idx]: (Seq_Len, Num_Nodes, Features)
        # Sum over seq_len (or last few) and features
        input_seq = inputs[idx] # (60, N, k)
        # Take last 6 steps
        recent_input = input_seq[-6:].sum(axis=(0, 2)) # (N,)
        
        target_map = targets[idx].sum(axis=1) # (N,) - Sum over k (time intervals in target)
        pred_map = preds[idx].sum(axis=1) # (N,) - Sum of probs
        
        # Reshape to grid
        # Note: grid_idx = lat_idx * lng_steps + lng_idx
        # So reshape needs to be (lat_steps, lng_steps)
        
        def to_grid(arr):
            # arr is (N,)
            # If N != lat*lng, we might have issues if mask was applied?
            # But preprocess doesn't seem to drop nodes, just fills with 0.
            # However, preprocess calculates num_grids = lat * lng.
            # So reshape is safe.
            return arr.reshape(lat_steps, lng_steps)
            
        grid_input = to_grid(recent_input)
        grid_target = to_grid(target_map)
        grid_pred = to_grid(pred_map)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot Input
        sns.heatmap(grid_input, ax=axes[0], cmap='viridis', cbar=True)
        axes[0].set_title(f'Recent Input Demand (Last 6 steps)')
        axes[0].invert_yaxis() # Map usually has 0 at bottom
        
        # Plot Target
        sns.heatmap(grid_target, ax=axes[1], cmap='Blues', cbar=True)
        axes[1].set_title(f'Ground Truth Demand')
        axes[1].invert_yaxis()
        
        # Plot Prediction
        sns.heatmap(grid_pred, ax=axes[2], cmap='Reds', cbar=True)
        axes[2].set_title(f'Predicted Probability Sum')
        axes[2].invert_yaxis()
        
        plt.suptitle(f'{case_type} Case #{idx} - Model: {model_name}')
        plt.tight_layout()
        plt.savefig(f'{save_dir}/{case_type}_sample_{idx}.png')
        plt.close()

    for idx in top_fp_indices:
        plot_sample(idx, 'FalsePositive')
        
    for idx in top_fn_indices:
        plot_sample(idx, 'FalseNegative')
        
    logging.info(f"Saved visualization analysis to {save_dir}")

def plot_prediction_results(json_path):
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        logging.error(f"Failed to load {json_path}: {e}")
        return
    os.makedirs('results/plots', exist_ok=True)
    sns.set(style="whitegrid")
    for dataset_name, per_dt in data.items():
        dts = sorted(per_dt.keys(), key=lambda x: int(x))
        algorithms = set()
        for dt in dts:
            for algo in per_dt[dt].keys():
                algorithms.add(algo)
        def collect(metric):
            series = {}
            for algo in algorithms:
                xs = []
                ys = []
                for dt in dts:
                    if algo in per_dt[dt]:
                        xs.append(int(dt))
                        ys.append(per_dt[dt][algo].get(metric, None))
                series[algo] = (xs, ys)
            return series
        for metric, ylabel, fname in [
            ('AP', 'Average Precision', f'results/plots/{dataset_name}_AP.png'),
            ('TrainTime', 'Train Time (s)', f'results/plots/{dataset_name}_TrainTime.png'),
            ('TestTime', 'Test Time (s)', f'results/plots/{dataset_name}_TestTime.png'),
        ]:
            plt.figure(figsize=(7,5))
            series = collect(metric)
            for algo, (xs, ys) in series.items():
                if len(xs) > 0:
                    sns.lineplot(x=xs, y=ys, label=algo, marker='o')
            plt.xlabel('dT (seconds)')
            plt.ylabel(ylabel)
            plt.title(f'{dataset_name} - {metric} vs dT')
            plt.legend()
            plt.tight_layout()
            plt.savefig(fname)
            plt.close()
def run_experiments():
    data_dir = 'data/raw'
    results = {}
    
    # Setup TensorBoard: tensorboard --logdir runs/prediction_experiment --port 6006 --bind_all
    writer = SummaryWriter('runs/prediction_experiment')
    
    quick = os.getenv('QUICK') == '1'
    paper_mode = True
    
    delta_ts = [5] if quick else [5, 6, 7, 8, 9]
    
    if paper_mode:
        logging.info("Running in PAPER STRICT mode (Specific Hours Only)")
        # Paper Time Ranges (UTC+8 for Chengdu)
        # Yueche: Experiment 9:00-11:00 (1:00-3:00 UTC). History: Preceding hour 8:00-9:00 (0:00-1:00 UTC).
        # Total Yueche: 8:00-11:00 Local -> 0:00-3:00 UTC.
        # DiDi: Experiment 21:00-23:00 (13:00-15:00 UTC). History: Preceding hour 20:00-21:00 (12:00-13:00 UTC).
        # Total DiDi: 20:00-23:00 Local -> 12:00-15:00 UTC.
        datasets = [
            {'name': 'Chengdu_Nov01_Paper_Yueche', 'worker': 'CN01_W', 'request': 'CN01_R', 'date': '2016-11-01', 'start_hour': 0, 'end_hour': 3},
            {'name': 'Chengdu_Nov01_Paper_DiDi', 'worker': 'CN01_W', 'request': 'CN01_R', 'date': '2016-11-01', 'start_hour': 12, 'end_hour': 15},
        ]
        # For Paper Mode, we need to adjust splits to ensure Training covers the first hour (History)
        # and Testing covers the experiment period.
        # Total 3 hours. 1st hour is History (Train). Next 2 hours are Experiment (Test).
        # Train Split should be 1/3 ~ 0.333.
        # However, we also need Validation.
        CONFIG['train_split'] = 0.34
        CONFIG['val_split'] = 0.01 # Minimal validation
        CONFIG['test_split'] = 0.65
        
    else:
        logging.info("Running in FULL DAY mode (00:00 - 24:00)")
        # Datasets with full day coverage (00:00 - 24:00 Local Time)
        # UTC: Previous Day 16:00 to Current Day 16:00 (-8 to 16 relative to Date 00:00 UTC)
        datasets = [
            {'name': 'Chengdu_Nov01_FullDay', 'worker': 'CN01_W', 'request': 'CN01_R', 'date': '2016-11-01', 'start_hour': -8, 'end_hour': 16},
            {'name': 'Chengdu_Nov15_FullDay', 'worker': 'CN15_W', 'request': 'CN15_R', 'date': '2016-11-15', 'start_hour': -8, 'end_hour': 16},
        ]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    
    for dataset_info in datasets:
        dataset_name = dataset_info['name']
        results[dataset_name] = {}
        
        for delta_t in delta_ts:
            logging.info(f"Processing {dataset_name}, Delta T = {delta_t}")
            print(f"\n{'='*30}\nProcessing {dataset_name}, Delta T = {delta_t}\n{'='*30}")
            
            # Preprocess
            logging.info("Preprocessing data...")
            # Check for GRID settings
            grid_mode = os.getenv('GRID_MODE', '1km') # '1km' or '0.02deg'
            if grid_mode == '0.02deg':
                logging.info("Using 0.02 degree grid (Coarse)")
                preprocessor = Preprocessor(data_dir, grid_size=0.02, grid_size_m=None, delta_t=delta_t)
            else:
                logging.info("Using 1km grid (Fine)")
                preprocessor = Preprocessor(data_dir, grid_size_m=1000, delta_t=delta_t)
            
            data_dict = preprocessor.process(dataset_info['worker'], dataset_info['request'], dataset_info['start_hour'], dataset_info['end_hour'], dataset_info['date'])
            C = data_dict['C']
            # S = data_dict['S'] # Not used
            grid_shape = data_dict['grid_shape']
            adj_mask = data_dict['adj_mask']
            adj_tensor = torch.FloatTensor(adj_mask).to(device)
            
            # Use only Demand (C) as per paper
            # C: (T, N, k) -> data: (T, N, k)
            data = C
            
            # Use previous hour to predict next hour by default; allow STRICT_P to fix P
            k = C.shape[2]
            vec_span = delta_t * k
            vectors_per_hour = max(1, int(np.ceil(3600 / vec_span)))
            strict_p = os.getenv('STRICT_P') == '1'
            seq_len = CONFIG['seq_len'] if strict_p else vectors_per_hour
            
            total_len = len(data)
            
            # Split 8:1:1 without overlap
            # Ensure we have enough data for seq_len context in Val and Test
            
            # Indices:
            # Train: [0, train_end]
            # Val: [train_end, val_end] (Needs context from Train)
            # Test: [val_end, total_len] (Needs context from Val)
            
            # Effective samples:
            # Train samples: data[0:train_end] -> (X, Y)
            # Val samples: data[train_end:val_end] -> (X, Y)
            # Test samples: data[val_end:total_len] -> (X, Y)
            
            # To generate X for Val at index `train_end`, we need data[train_end - seq_len : train_end]
            # So val_data slice should start earlier.
            
            num_samples = total_len - seq_len
            train_cnt = int(num_samples * 0.8)
            val_cnt = int(num_samples * 0.1)
            test_cnt = num_samples - train_cnt - val_cnt
            
            # Data Slices
            # Train
            train_start_idx = 0
            train_end_idx = train_cnt + seq_len
            train_data = data[train_start_idx:train_end_idx]
            
            # Val
            val_start_idx = train_cnt
            val_end_idx = train_cnt + val_cnt + seq_len
            val_data = data[val_start_idx:val_end_idx]
            
            # Test
            test_start_idx = train_cnt + val_cnt
            test_end_idx = total_len # Should be equal to test_start_idx + test_cnt + seq_len
            test_data = data[test_start_idx:test_end_idx]
            
            logging.info(f"Split sizes (samples): Train={train_cnt}, Val={val_cnt}, Test={test_cnt}")
            
            # Loss configuration
            use_strict_loss = os.getenv('STRICT_LOSS') == '1'
            train_demand = train_data
            if use_strict_loss:
                pos_weight = None
                logging.info("Using strict BCEWithLogitsLoss without focal or sample weighting")
            else:
                num_pos = train_demand.sum()
                num_neg = train_demand.size - num_pos
                pos_weight_val = num_neg / (num_pos + 1e-5)
                pos_weight_val = min(pos_weight_val, 50.0)
                pos_weight = torch.tensor([pos_weight_val]).to(device)
                logging.info(f"Positive Rate: {num_pos/train_demand.size:.6f}, Positive Weight: {pos_weight_val:.2f}")
            node_activity = train_demand.sum(axis=(0,2))
            node_activity_mean = float(np.mean(node_activity)) if node_activity.size > 0 else 1.0
            node_weights_np = node_activity / (node_activity_mean + 1e-5)
            node_weights_np = np.clip(node_weights_np, 0.2, 5.0)
            node_weights = torch.tensor(node_weights_np, dtype=torch.float32).to(device)
            
            
            # Create Datasets (Input: D+S, Output: D)
            X_train_full, Y_train_full = create_dataset(train_data, seq_len)
            X_val_full, Y_val_full = create_dataset(val_data, seq_len)
            X_test_full, Y_test_full = create_dataset(test_data, seq_len)
            
            # Slice Y to be Demand only (Y is already Demand only since data=C)
            Y_train = Y_train_full
            Y_val = Y_val_full
            Y_test = Y_test_full
            
            # X keeps both
            X_train = X_train_full
            X_val = X_val_full
            X_test = X_test_full
            
            train_dataset = TensorDataset(X_train, Y_train)
            val_dataset = TensorDataset(X_val, Y_val)
            test_dataset = TensorDataset(X_test, Y_test)
            gen = torch.Generator(); gen.manual_seed(SEED)
            
            num_nodes = data.shape[1]
            input_dim = data.shape[2] # k
            output_dim = k # Predict Demand only
            
            models = {
                'LSTM': LSTMModel(input_dim, 256, output_dim, num_layers=2).to(device),
                'GraphWaveNet': GraphWaveNet(num_nodes, input_dim, in_dim=input_dim, out_dim=output_dim, device=device, supports=[adj_tensor], blocks=5, layers=3).to(device),
                'DDGNN': DDGNN(num_nodes, input_dim, 128, output_dim, seq_len, adj_prior=adj_tensor, tcn_depth=6, appnp_k=8).to(device)
            }
            
            results[dataset_name][delta_t] = {}
            
            for model_name, model in models.items():
                logging.info(f"Training {model_name}...")
                criterion = (nn.BCEWithLogitsLoss() if use_strict_loss
                             else FocalBCEWithLogitsLoss(alpha=1.0, gamma=2.0, pos_weight=pos_weight, node_weights=node_weights))
                optimizer = optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=1e-5)
                
                tag_prefix = f"{dataset_name}/dT{delta_t}"
                
                bs = CONFIG['batch_size']
                while True:
                    try:
                        if use_strict_loss:
                            train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, generator=gen)
                        else:
                            sample_pos = (Y_train.sum(dim=(1,2)) > 0).float()
                            weights = torch.where(sample_pos > 0, torch.tensor(5.0), torch.tensor(1.0))
                            sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples=len(weights), replacement=True, generator=gen)
                            train_loader = DataLoader(train_dataset, batch_size=bs, sampler=sampler, generator=gen)
                        val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False, generator=gen)
                        test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False, generator=gen)
                        train_time = train_model(model, train_loader, val_loader, criterion, optimizer, device, writer, tag_prefix, model_name, CONFIG)
                        ap, test_time, preds, targets, inputs = evaluate_model(model, test_loader, device)
                        break
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        new_bs = max(1, bs // 2)
                        if new_bs == bs:
                            raise
                        bs = new_bs
                        logging.warning(f"OOM encountered, reducing batch size to {bs}")
                if os.getenv('STRICT_AP') == '1':
                    # Macro AP over nodes
                    preds_np = preds
                    targets_np = targets
                    num_nodes_eval = preds_np.shape[1]
                    node_aps = []
                    for node_idx in range(num_nodes_eval):
                        p = preds_np[:, node_idx, :].reshape(-1)
                        t = targets_np[:, node_idx, :].reshape(-1)
                        try:
                            node_ap = average_precision_score(t, p)
                        except Exception:
                            node_ap = 0.0
                        node_aps.append(node_ap)
                    ap_macro = float(np.mean(node_aps))
                    ap = ap_macro
                
                logging.info(f"Finished {model_name} - AP: {ap:.4f}, Train Time: {train_time:.2f}s, Test Time: {test_time:.2f}s")
                writer.add_scalar(f'{tag_prefix}/Metrics/AP/{model_name}', ap, 0)
                writer.add_scalar(f'{tag_prefix}/Time/Train/{model_name}', train_time, 0)
                writer.add_scalar(f'{tag_prefix}/Time/Test/{model_name}', test_time, 0)
                
                # Save Raw Predictions for Visualization App
                raw_save_dir = f'results/raw_preds/{dataset_name}_dT{delta_t}/{model_name}'
                os.makedirs(raw_save_dir, exist_ok=True)
                np.save(f'{raw_save_dir}/preds.npy', preds)
                np.save(f'{raw_save_dir}/targets.npy', targets)
                # Save inputs if needed, but they can be large
                # np.save(f'{raw_save_dir}/inputs.npy', inputs)
                logging.info(f"Saved raw predictions to {raw_save_dir}")

                # Visualize bad cases
                visualize_cases(model_name, inputs, preds, targets, grid_shape, dataset_name, delta_t)
                
                results[dataset_name][delta_t][model_name] = {
                    'AP': ap,
                    'TrainTime': train_time,
                    'TestTime': test_time
                }
                
    # Save results
    writer.close()
    os.makedirs('results', exist_ok=True)
    with open('results/prediction_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    logging.info("Results saved to results/prediction_results.json")
    plot_prediction_results('results/prediction_results.json')

if __name__ == "__main__":
    run_experiments()
