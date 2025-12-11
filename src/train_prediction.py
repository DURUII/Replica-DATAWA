import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import numpy as np
from model import DDGNN
from torch.utils.data import DataLoader, TensorDataset
import os

def create_dataset(data, seq_len):
    # data: (Total_Time, Num_Nodes, k)
    X, Y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        Y.append(data[i+seq_len])
    return torch.FloatTensor(np.array(X)), torch.FloatTensor(np.array(Y))

def train():
    # Load data
    with open('data/processed/yueche_data.pkl', 'rb') as f:
        data_dict = pickle.load(f)
    
    C = data_dict['C'] # (720, 270, 3)
    
    # Split
    # Train on first hour (240 steps)
    train_len = 240
    train_data = C[:train_len]
    
    seq_len = 12
    X_train, Y_train = create_dataset(train_data, seq_len)
    
    print(f"Train shapes: X {X_train.shape}, Y {Y_train.shape}")
    
    # Model
    num_nodes = C.shape[1]
    input_dim = C.shape[2]
    output_dim = input_dim
    
    model = DDGNN(num_nodes, input_dim, 64, output_dim, seq_len)
    
    # Training
    criterion = nn.MSELoss() # Or BCEWithLogitsLoss if binary?
    # The paper says "binary value (1 for yes, 0 for no)".
    # But Eq 193 says "If c... exceeds a given threshold (i.e, 0.85)".
    # So we should predict probabilities.
    # So BCEWithLogitsLoss is better.
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    dataset = TensorDataset(X_train, Y_train)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    model.train()
    for epoch in range(20):
        total_loss = 0
        for x, y in loader:
            optimizer.zero_grad()
            out, adj = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader):.4f}")
        
    # Save model
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'models/ddgnn_yueche.pth')
    print("Model saved.")

if __name__ == "__main__":
    train()
