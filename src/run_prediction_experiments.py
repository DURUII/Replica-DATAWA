import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
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

def create_dataset(data, seq_len):
    # data: (Total_Time, Num_Nodes, k)
    X, Y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        Y.append(data[i+seq_len])
    return torch.FloatTensor(np.array(X)), torch.FloatTensor(np.array(Y))

def train_model(model, train_loader, criterion, optimizer, device, writer, tag_prefix, epochs=20):
    model.train()
    start_time = time.time()
    
    global_step = 0
    steps_per_epoch = len(train_loader)
    
    # Outer progress bar for epochs
    epoch_pbar = tqdm(range(epochs), desc=f"Training {tag_prefix}", position=0, leave=True)
    
    for epoch in epoch_pbar:
        total_loss = 0
        num_batches = 0
        
        # Inner progress bar for batches
        # batch_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}", position=1, leave=False)
        
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
            optimizer.step()
            
            current_loss = loss.item()
            total_loss += current_loss
            num_batches += 1
            
            # Log step loss
            global_step_val = epoch * steps_per_epoch + num_batches
            writer.add_scalar(f'{tag_prefix}/StepLoss', current_loss, global_step_val)
            
            # Update batch pbar
            # batch_pbar.set_postfix({'loss': f"{current_loss:.4f}"})
            
        avg_loss = total_loss / num_batches
        writer.add_scalar(f'{tag_prefix}/EpochLoss', avg_loss, epoch)
        
        # Update epoch pbar
        epoch_pbar.set_postfix({'AvgLoss': f"{avg_loss:.4f}"})
        logging.info(f"[{tag_prefix}] Epoch {epoch+1}/{epochs} - Avg Loss: {avg_loss:.4f}")
        
    end_time = time.time()
    return end_time - start_time

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
    
    # Setup TensorBoard
    writer = SummaryWriter('runs/prediction_experiment')
    
    delta_ts = [5, 6, 7, 8, 9]
    datasets = [
        {'name': 'Yueche', 'start_hour': 0, 'end_hour': 3}, # 8:00 - 11:00 local (0-3 UTC)
        {'name': 'DiDi', 'start_hour': 12, 'end_hour': 15}  # 20:00 - 23:00 local (12-15 UTC)
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
            preprocessor = Preprocessor(data_dir, delta_t=delta_t)
            # Use CN01_R/W for both as discussed
            data_dict = preprocessor.process('CN01_W', 'CN01_R', dataset_info['start_hour'], dataset_info['end_hour'], '2016-11-01')
            C = data_dict['C']
            
            # Split 80/20
            total_len = len(C)
            train_len = int(total_len * 0.8)
            train_data = C[:train_len]
            test_data = C[train_len:]
            
            seq_len = 12 
            
            X_train, Y_train = create_dataset(train_data, seq_len)
            X_test, Y_test = create_dataset(test_data, seq_len)
            
            train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=32, shuffle=True)
            test_loader = DataLoader(TensorDataset(X_test, Y_test), batch_size=32, shuffle=False)
            
            num_nodes = C.shape[1]
            input_dim = C.shape[2]
            output_dim = input_dim
            
            models = {
                'LSTM': LSTMModel(input_dim, 64, output_dim).to(device),
                'GraphWaveNet': GraphWaveNet(num_nodes, input_dim, out_dim=output_dim).to(device),
                'DDGNN': DDGNN(num_nodes, input_dim, 64, output_dim, seq_len).to(device)
            }
            
            results[dataset_name][delta_t] = {}
            
            for model_name, model in models.items():
                logging.info(f"Training {model_name}...")
                criterion = nn.BCEWithLogitsLoss()
                optimizer = optim.Adam(model.parameters(), lr=0.001)
                
                tag_prefix = f"{dataset_name}/dT{delta_t}/{model_name}"
                
                train_time = train_model(model, train_loader, criterion, optimizer, device, writer, tag_prefix)
                ap, test_time = evaluate_model(model, test_loader, device)
                
                logging.info(f"Finished {model_name} - AP: {ap:.4f}, Train Time: {train_time:.2f}s, Test Time: {test_time:.2f}s")
                writer.add_scalar(f'{tag_prefix}/AP', ap, 0)
                writer.add_scalar(f'{tag_prefix}/TrainTime', train_time, 0)
                writer.add_scalar(f'{tag_prefix}/TestTime', test_time, 0)
                
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
