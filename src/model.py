import torch
import torch.nn as nn
import torch.nn.functional as F

class DemandDependencyLearning(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_nodes):
        super(DemandDependencyLearning, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(input_dim, hidden_dim)
        self.num_nodes = num_nodes

    def forward(self, x):
        # x: (Batch, Num_Nodes, Input_Dim)
        # We want to learn A based on the features at the current time step?
        # Or is A learned globally?
        # The paper says "dynamic time-based adjacency matrix A^t".
        # So it depends on x.
        
        m1 = self.fc1(x) # (Batch, Num_Nodes, Hidden)
        m2 = self.fc2(x) # (Batch, Num_Nodes, Hidden)
        
        # A = Softmax(Tanh(M1 M2^T + M2 M1^T))
        # We need to handle batch dimension
        # m1: (B, N, H), m2: (B, N, H)
        # m1 @ m2.T -> (B, N, N)
        
        adj = torch.matmul(m1, m2.transpose(1, 2)) + torch.matmul(m2, m1.transpose(1, 2))
        adj = torch.tanh(adj)
        adj = F.softmax(adj, dim=-1)
        return adj

class DilatedCausalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super(DilatedCausalConv, self).__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)
        self.conv2 = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)
        
    def forward(self, x):
        # x: (Batch, Channels, Seq_Len)
        # Pad to keep length same (causal)
        x_pad = F.pad(x, (self.pad, 0))
        
        filter = torch.tanh(self.conv1(x_pad))
        gate = torch.sigmoid(self.conv2(x_pad))
        
        return filter * gate

class APPNP(nn.Module):
    def __init__(self, k=10, alpha=0.1):
        super(APPNP, self).__init__()
        self.k = k
        self.alpha = alpha

    def forward(self, x, adj):
        # x: (Batch, Num_Nodes, Features)
        # adj: (Batch, Num_Nodes, Num_Nodes)
        
        # Normalize adj
        # A_hat = D^-1/2 (A + I) D^-1/2
        # Here adj is already Softmaxed, so it's row-normalized?
        # The paper says A^t = Softmax(...).
        # Then A_hat = ...
        # Let's assume adj is the raw A^t.
        
        # Add self loop
        batch_size, num_nodes, _ = adj.shape
        eye = torch.eye(num_nodes).to(adj.device).unsqueeze(0).expand(batch_size, -1, -1)
        adj = adj + eye
        
        # Degree matrix
        d = adj.sum(dim=-1) # (B, N)
        d_inv_sqrt = d.pow(-0.5)
        d_inv_sqrt[d_inv_sqrt == float('inf')] = 0
        d_mat = torch.diag_embed(d_inv_sqrt) # (B, N, N)
        
        norm_adj = torch.matmul(torch.matmul(d_mat, adj), d_mat)
        
        z = x
        for _ in range(self.k):
            # Z = alpha * x + (1-alpha) * A * Z
            z = self.alpha * x + (1 - self.alpha) * torch.matmul(norm_adj, z)
            # Paper says ReLU at the end?
            # Eq 188: Z^H = ReLU(...)
            # Intermediate steps: Eq 185 doesn't have ReLU.
        
        return F.relu(z)

class DDGNN(nn.Module):
    def __init__(self, num_nodes, input_dim, hidden_dim, output_dim, seq_len, kernel_size=3, dilation_channels=32):
        super(DDGNN, self).__init__()
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.seq_len = seq_len
        
        # Demand Dependency Learning
        self.ddl = DemandDependencyLearning(input_dim, hidden_dim, num_nodes)
        
        # Dilated Causal Conv
        # Input to conv is (Batch, Input_Dim, Seq_Len) for each node?
        # Or do we treat nodes as channels?
        # Usually in ST-GNN, we process (B, N, C, T).
        # We can reshape to (B*N, C, T).
        self.tcn = DilatedCausalConv(input_dim, dilation_channels, kernel_size=kernel_size, dilation=1)
        # Stack more layers if needed, but paper mentions one layer or doesn't specify depth.
        # "increasing the layer depth"
        # Let's add a few layers with increasing dilation.
        self.tcn_layers = nn.ModuleList([
            DilatedCausalConv(input_dim, dilation_channels, kernel_size, dilation=1),
            DilatedCausalConv(dilation_channels, dilation_channels, kernel_size, dilation=2),
            DilatedCausalConv(dilation_channels, dilation_channels, kernel_size, dilation=4)
        ])
        
        # APPNP
        self.appnp = APPNP(k=10, alpha=0.1)
        
        # Output layer
        self.fc_out = nn.Linear(dilation_channels, output_dim)

    def forward(self, x):
        # x: (Batch, Seq_Len, Num_Nodes, Input_Dim)
        batch_size, seq_len, num_nodes, input_dim = x.shape
        
        # 1. Demand Dependency Learning
        # We use the features at the last time step to learn A?
        # Or average over time?
        # Paper says "from historical task data C^t at time instance t".
        # It seems A is dynamic per time step.
        # But if we predict the NEXT step, we might want to use the most recent info.
        # Let's use the last time step x[:, -1, :, :]
        adj = self.ddl(x[:, -1, :, :]) # (B, N, N)
        
        # 2. TCN
        # Reshape to (B*N, Input_Dim, Seq_Len)
        x_tcn = x.permute(0, 2, 3, 1).reshape(batch_size * num_nodes, input_dim, seq_len)
        
        skip = 0
        for layer in self.tcn_layers:
            x_tcn = layer(x_tcn)
            
        # Take the last time step of TCN output
        x_tcn = x_tcn[:, :, -1] # (B*N, Channels)
        x_tcn = x_tcn.reshape(batch_size, num_nodes, -1) # (B, N, Channels)
        
        # 3. APPNP
        z = self.appnp(x_tcn, adj) # (B, N, Channels)
        
        # 4. Output
        out = self.fc_out(z) # (B, N, Output_Dim)
        
        return out, adj
