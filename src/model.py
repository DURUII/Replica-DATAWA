import torch
import torch.nn as nn
import torch.nn.functional as F

class DemandDependencyLearning(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_nodes):
        super(DemandDependencyLearning, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(input_dim, hidden_dim)
        self.node_emb1 = nn.Parameter(torch.randn(num_nodes, hidden_dim))
        self.node_emb2 = nn.Parameter(torch.randn(num_nodes, hidden_dim))
        self.num_nodes = num_nodes
        self.act = nn.ELU()
        self.dropout = nn.Dropout(0.2)
        self.temperature = 0.7

    def forward(self, x):
        # x: (Batch, Num_Nodes, Input_Dim)
        
        m1 = self.dropout(self.act(self.fc1(x))) + self.node_emb1.unsqueeze(0) # (Batch, Num_Nodes, Hidden)
        m2 = self.dropout(self.act(self.fc2(x))) + self.node_emb2.unsqueeze(0) # (Batch, Num_Nodes, Hidden)
        
        # A = Softmax(Tanh(M1 M2^T + M2 M1^T))
        
        adj_scores = torch.matmul(m1, m2.transpose(1, 2)) + torch.matmul(m2, m1.transpose(1, 2))
        adj_scores = torch.tanh(adj_scores)
        adj = F.softmax(adj_scores / self.temperature, dim=-1)
        return adj

class DilatedCausalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super(DilatedCausalConv, self).__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)
        self.conv2 = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)
        
        # 1x1 conv for residual if channels change
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # x: (Batch, Channels, Seq_Len)
        # Pad to keep length same (causal)
        x_pad = F.pad(x, (self.pad, 0))
        
        filter = torch.tanh(self.conv1(x_pad))
        gate = torch.sigmoid(self.conv2(x_pad))
        
        out = filter * gate
        out = self.dropout(out)
        
        if self.residual_conv is not None:
            res = self.residual_conv(x)
        else:
            res = x
            
        return out + res

class APPNP(nn.Module):
    def __init__(self, k=5, alpha=0.2, dropout=0.2):
        super(APPNP, self).__init__()
        self.k = k
        self.alpha = alpha
        self.dropout = dropout

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
            z = F.dropout(z, p=self.dropout, training=self.training)
            # Paper says ReLU at the end?
            # Eq 188: Z^H = ReLU(...)
            # Intermediate steps: Eq 185 doesn't have ReLU.
        
        return F.relu(z)

class DDGNN(nn.Module):
    def __init__(self, num_nodes, input_dim, hidden_dim, output_dim, seq_len, kernel_size=3, dilation_channels=32, node_emb_dim=10, adj_prior=None, tcn_depth=6, appnp_k=8):
        super(DDGNN, self).__init__()
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.decay_lambda = 0.5
        self.adj_prior = adj_prior
        self.tcn_depth = tcn_depth
        
        # Node Identity Embedding
        self.node_emb = nn.Parameter(torch.randn(num_nodes, node_emb_dim))
        
        # Update input dim for internal layers
        self.combined_input_dim = input_dim + node_emb_dim
        
        # Demand Dependency Learning
        self.ddl = DemandDependencyLearning(self.combined_input_dim, hidden_dim, num_nodes)
        
        # Dilated Causal Conv
        # Input to conv is (Batch, Input_Dim, Seq_Len) for each node?
        # Or do we treat nodes as channels?
        # Usually in ST-GNN, we process (B, N, C, T).
        # We can reshape to (B*N, C, T).
        
        # Stack more layers if needed, but paper mentions one layer or doesn't specify depth.
        # "increasing the layer depth"
        # Let's add a few layers with increasing dilation.
        layers = []
        d = 1
        for i in range(self.tcn_depth):
            in_ch = self.combined_input_dim if i == 0 else dilation_channels
            layers.append(DilatedCausalConv(in_ch, dilation_channels, kernel_size, dilation=d))
            d = d * 2
        self.tcn_layers = nn.ModuleList(layers)
        
        self.appnp = APPNP(k=appnp_k, alpha=0.2, dropout=0.2)
        
        # Output head
        self.head = nn.Sequential(
            nn.Linear(dilation_channels, dilation_channels),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(dilation_channels, output_dim)
        )

    def forward(self, x):
        # x: (Batch, Seq_Len, Num_Nodes, Input_Dim)
        batch_size, seq_len, num_nodes, input_dim = x.shape
        
        # Add Node Embeddings
        # node_emb: (Num_Nodes, Emb_Dim) -> (1, 1, Num_Nodes, Emb_Dim) -> (B, T, N, Emb_Dim)
        node_emb = self.node_emb.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, num_nodes, -1)
        x_combined = torch.cat([x, node_emb], dim=-1) # (B, T, N, Input+Emb)
        
        # 1. Demand Dependency Learning
        # Weighted temporal aggregation for adjacency (emphasize recent steps)
        with torch.no_grad():
            steps = torch.arange(seq_len, device=x.device)
            weights = torch.exp(-self.decay_lambda * (seq_len - 1 - steps)).view(1, seq_len, 1, 1)
            weights = weights / weights.sum()
        
        # Use combined features for DDL to include node identity
        x_adj = (x_combined * weights).sum(dim=1)  # (B, N, C_combined)
        adj = self.ddl(x_adj) # (B, N, N)
        if self.adj_prior is not None:
            ap = self.adj_prior.to(adj.device)
            ap = ap.unsqueeze(0).expand(batch_size, -1, -1)
            adj = 0.5 * adj + 0.5 * ap
            adj = adj / adj.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        
        # 2. TCN
        # Reshape to (B*N, Input_Dim, Seq_Len)
        # Permute: (B, T, N, C) -> (B, N, C, T) -> (B*N, C, T)
        x_tcn = x_combined.permute(0, 2, 3, 1).reshape(batch_size * num_nodes, self.combined_input_dim, seq_len)
        
        skip = 0
        for layer in self.tcn_layers:
            x_tcn = layer(x_tcn)
            
        # Take the last time step of TCN output
        x_tcn = x_tcn[:, :, -1] # (B*N, Channels)
        x_tcn = x_tcn.reshape(batch_size, num_nodes, -1) # (B, N, Channels)
        
        # 3. APPNP
        z = self.appnp(x_tcn, adj) # (B, N, Channels)
        z = z + x_tcn  # residual fusion to keep local temporal features
        
        # 4. Output
        out = self.head(z) # (B, N, Output_Dim)
        
        return out, adj
