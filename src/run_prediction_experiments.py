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
from tqdm import tqdm
from sklearn.metrics import average_precision_score
from preprocess import Preprocessor
from model import DDGNN
from baselines import LSTMModel, GraphWaveNet

# Setup logging
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
    'learning_rate': 0.01,
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
    def __init__(self, alpha=1.0, gamma=2.0, pos_weight=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=self.pos_weight, reduction='none')
        probs = torch.sigmoid(logits)
        pt = probs * targets + (1 - probs) * (1 - targets)
        loss = self.alpha * (1 - pt) ** self.gamma * bce
        return loss.mean()
def train_model(model, train_loader, val_loader, criterion, optimizer, device, writer, tag_prefix, model_name_only, config):
    model.train()
    start_time = time.time()
    
    global_step = 0
    steps_per_epoch = len(train_loader)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Outer progress bar for epochs
    epoch_pbar = tqdm(range(config['epochs']), desc=f"Training {tag_prefix}", position=0, leave=True)
    
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
        
        # Update epoch pbar
        epoch_pbar.set_postfix({'TrainLoss': f"{avg_loss:.4f}", 'ValLoss': f"{val_loss:.4f}"})
        logging.info(f"[{tag_prefix}] Epoch {epoch+1}/{config['epochs']} - Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model state? For now just continue
        else:
            patience_counter += 1
            if patience_counter >= config['patience']:
                logging.info(f"Early stopping at epoch {epoch+1}")
                break
        
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
    start_time = time.time()
    with torch.no_grad():
        for x, y in test_loader:
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
    end_time = time.time()
    
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    
    # Flatten for AP calculation
    ap = average_precision_score(all_targets.flatten(), all_preds.flatten())
    return ap, end_time - start_time

def run_experiments():
    data_dir = 'data/raw'
    results = {}
    
    # Setup TensorBoard: tensorboard --logdir runs/prediction_experiment --port 6006 --bind_all
    writer = SummaryWriter('runs/prediction_experiment')
    
    quick = os.getenv('QUICK') == '1'
    delta_ts = [5] if quick else [5, 6, 7, 8, 9]
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
            preprocessor = Preprocessor(data_dir, grid_size=0.02, delta_t=delta_t)
            
            data_dict = preprocessor.process(dataset_info['worker'], dataset_info['request'], dataset_info['start_hour'], dataset_info['end_hour'], dataset_info['date'])
            C = data_dict['C']
            S = data_dict['S']
            
            # Concatenate Demand (C) and Supply (S)
            # C: (T, N, k), S: (T, N, k) -> data: (T, N, 2k)
            data = np.concatenate([C, S], axis=-1)
            
            # Split Train/Val/Test (70/10/20)
            total_len = len(data)
            train_len = int(total_len * CONFIG['train_split'])
            val_len = int(total_len * CONFIG['val_split'])
            test_len = total_len - train_len - val_len
            
            train_data = data[:train_len]
            val_data = data[train_len:train_len+val_len]
            test_data = data[train_len+val_len:]
            
            # Calculate pos_weight based on DEMAND only (first k channels)
            k = C.shape[2]
            train_demand = train_data[:, :, :k]
            num_pos = train_demand.sum()
            num_neg = train_demand.size - num_pos
            pos_weight_val = num_neg / (num_pos + 1e-5)
            pos_weight_val = min(pos_weight_val, 50.0)
            pos_weight = torch.tensor([pos_weight_val]).to(device)
            logging.info(f"Positive Rate: {num_pos/train_demand.size:.6f}, Positive Weight: {pos_weight_val:.2f}")
            
            seq_len = CONFIG['seq_len']
            
            # Create Datasets (Input: D+S, Output: D)
            X_train_full, Y_train_full = create_dataset(train_data, seq_len)
            X_val_full, Y_val_full = create_dataset(val_data, seq_len)
            X_test_full, Y_test_full = create_dataset(test_data, seq_len)
            
            # Slice Y to be Demand only
            Y_train = Y_train_full[:, :, :k]
            Y_val = Y_val_full[:, :, :k]
            Y_test = Y_test_full[:, :, :k]
            
            # X keeps both
            X_train = X_train_full
            X_val = X_val_full
            X_test = X_test_full
            
            train_dataset = TensorDataset(X_train, Y_train)
            sample_pos = (Y_train.sum(dim=(1,2)) > 0).float()
            weights = torch.where(sample_pos > 0, torch.tensor(3.0), torch.tensor(1.0))
            gen = torch.Generator()
            gen.manual_seed(SEED)
            sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples=len(weights), replacement=True, generator=gen)
            train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], sampler=sampler, generator=gen)
            val_loader = DataLoader(TensorDataset(X_val, Y_val), batch_size=CONFIG['batch_size'], shuffle=False, generator=gen)
            test_loader = DataLoader(TensorDataset(X_test, Y_test), batch_size=CONFIG['batch_size'], shuffle=False, generator=gen)
            
            num_nodes = data.shape[1]
            input_dim = data.shape[2] # 2k
            output_dim = k # Predict Demand only
            
            models = {
                'LSTM': LSTMModel(input_dim, 256, output_dim, num_layers=2, dropout=0.5).to(device),
                'GraphWaveNet': GraphWaveNet(num_nodes, input_dim, out_dim=output_dim).to(device),
                'DDGNN': DDGNN(num_nodes, input_dim, 128, output_dim, seq_len).to(device)
            }
            
            results[dataset_name][delta_t] = {}
            
            for model_name, model in models.items():
                logging.info(f"Training {model_name}...")
                criterion = FocalBCEWithLogitsLoss(alpha=1.0, gamma=2.0, pos_weight=pos_weight)
                optimizer = optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=1e-5)
                
                tag_prefix = f"{dataset_name}/dT{delta_t}"
                
                train_time = train_model(model, train_loader, val_loader, criterion, optimizer, device, writer, tag_prefix, model_name, CONFIG)
                ap, test_time = evaluate_model(model, test_loader, device)
                
                logging.info(f"Finished {model_name} - AP: {ap:.4f}, Train Time: {train_time:.2f}s, Test Time: {test_time:.2f}s")
                writer.add_scalar(f'{tag_prefix}/Metrics/AP/{model_name}', ap, 0)
                writer.add_scalar(f'{tag_prefix}/Time/Train/{model_name}', train_time, 0)
                writer.add_scalar(f'{tag_prefix}/Time/Test/{model_name}', test_time, 0)
                
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

if __name__ == "__main__":
    run_experiments()
