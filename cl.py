import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tqdm import tqdm

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# MRIFormer Components
class MultiResolutionEmbedding(nn.Module):
    """Extract multi-resolution representations"""
    def __init__(self, d_model, resolutions=[1, 2, 4, 8]):
        super().__init__()
        self.resolutions = resolutions
        self.projections = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in resolutions
        ])
        
    def forward(self, x):
        # x: [batch, seq_len, d_model]
        multi_res_features = []
        
        for i, res in enumerate(self.resolutions):
            if res == 1:
                feat = x
            else:
                # Average pooling for downsampling
                batch, seq_len, d_model = x.shape
                padded_len = ((seq_len + res - 1) // res) * res
                if padded_len != seq_len:
                    padding = torch.zeros(batch, padded_len - seq_len, d_model, device=x.device)
                    x_padded = torch.cat([x, padding], dim=1)
                else:
                    x_padded = x
                    
                feat = x_padded.reshape(batch, -1, res, d_model).mean(dim=2)
                # Upsample back to original length
                feat = feat.repeat_interleave(res, dim=1)[:, :seq_len, :]
            
            feat = self.projections[i](feat)
            multi_res_features.append(feat)
        
        return multi_res_features


class InteractionAttention(nn.Module):
    """Cross-resolution attention mechanism"""
    def __init__(self, d_model, n_heads=8, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.fc = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, q, k, v):
        batch_size = q.size(0)
        residual = q
        
        # Multi-head projections
        Q = self.w_q(q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(k).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(v).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        context = torch.matmul(attn, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        output = self.fc(context)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        
        return output


class MRIFormerBlock(nn.Module):
    """Multi-Resolution Interaction Block"""
    def __init__(self, d_model, n_heads=8, d_ff=2048, dropout=0.1, num_resolutions=4):
        super().__init__()
        self.num_resolutions = num_resolutions
        
        # Self-attention for each resolution
        self.self_attns = nn.ModuleList([
            InteractionAttention(d_model, n_heads, dropout)
            for _ in range(num_resolutions)
        ])
        
        # Cross-resolution attention
        self.cross_attns = nn.ModuleList([
            nn.ModuleList([
                InteractionAttention(d_model, n_heads, dropout)
                for _ in range(num_resolutions)
            ]) for _ in range(num_resolutions)
        ])
        
        # Feed-forward networks
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, d_model),
                nn.Dropout(dropout)
            ) for _ in range(num_resolutions)
        ])
        
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_resolutions)
        ])
        
    def forward(self, multi_res_features):
        # Self-attention at each resolution
        self_attn_outputs = []
        for i, feat in enumerate(multi_res_features):
            out = self.self_attns[i](feat, feat, feat)
            self_attn_outputs.append(out)
        
        # Cross-resolution interaction
        cross_attn_outputs = []
        for i in range(self.num_resolutions):
            cross_out = self_attn_outputs[i]
            for j in range(self.num_resolutions):
                if i != j:
                    cross_out = cross_out + self.cross_attns[i][j](
                        cross_out, self_attn_outputs[j], self_attn_outputs[j]
                    )
            cross_attn_outputs.append(cross_out)
        
        # Feed-forward
        outputs = []
        for i, feat in enumerate(cross_attn_outputs):
            ff_out = self.ffns[i](feat)
            out = self.layer_norms[i](ff_out + feat)
            outputs.append(out)
        
        return outputs


class MRIFormer(nn.Module):
    """Multi-Resolution Interaction Transformer"""
    def __init__(self, n_features, d_model=128, n_heads=8, n_layers=3, 
                 d_ff=512, dropout=0.1, seq_len=168, pred_len=1, resolutions=[1, 2, 4, 8]):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.d_model = d_model
        
        # Input embedding
        self.input_projection = nn.Linear(n_features, d_model)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, seq_len, d_model))
        
        # Multi-resolution embedding
        self.multi_res_embed = MultiResolutionEmbedding(d_model, resolutions)
        
        # MRIFormer blocks
        self.blocks = nn.ModuleList([
            MRIFormerBlock(d_model, n_heads, d_ff, dropout, len(resolutions))
            for _ in range(n_layers)
        ])
        
        # Fusion layer
        self.fusion = nn.Linear(d_model * len(resolutions), d_model)
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, pred_len)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: [batch, seq_len, n_features]
        batch_size = x.size(0)
        
        # Input embedding
        x = self.input_projection(x)
        x = x + self.pos_encoding
        x = self.dropout(x)
        
        # Multi-resolution embedding
        multi_res_features = self.multi_res_embed(x)
        
        # MRIFormer blocks
        for block in self.blocks:
            multi_res_features = block(multi_res_features)
        
        # Fusion
        fused = torch.cat(multi_res_features, dim=-1)
        fused = self.fusion(fused)
        
        # Use last time step for prediction
        output = self.output_projection(fused[:, -1, :])
        
        return output


class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_len=168, pred_len=1):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.data = data
        
    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1
    
    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_len]
        y = self.data[idx + self.seq_len:idx + self.seq_len + self.pred_len, 2]  # consumption column
        return torch.FloatTensor(x), torch.FloatTensor(y)


def prepare_data(df):
    """Prepare data with temporal features"""
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    
    # Extract temporal features
    df['hour'] = df['date'].dt.hour
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['day_of_year'] = df['date'].dt.dayofyear
    
    # Cyclical encoding for temporal features
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # Drop date and original temporal columns
    df = df.drop(['date', 'hour', 'day_of_week', 'month', 'day_of_year'], axis=1)
    
    return df


# Load and prepare data
print("Loading data...")
train_df = pd.read_csv('dataset/train.csv')
test_df = pd.read_csv('dataset/test.csv')

print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

# Prepare data
train_df = prepare_data(train_df)
test_df = prepare_data(test_df)

# Fill missing values
train_df = train_df.fillna(train_df.mean())
test_df = test_df.fillna(test_df.mean())

# Scale features
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_df.values)
test_scaled = scaler.transform(test_df.values)

# Hyperparameters
SEQ_LEN = 168  # 1 week of hourly data
PRED_LEN = 1   # Predict next hour
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.0001
D_MODEL = 128
N_HEADS = 8
N_LAYERS = 3
D_FF = 512
DROPOUT = 0.2  # Increased dropout to reduce overfitting

# Create datasets
train_dataset = TimeSeriesDataset(train_scaled, SEQ_LEN, PRED_LEN)
test_dataset = TimeSeriesDataset(test_scaled, SEQ_LEN, PRED_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Initialize model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

n_features = train_scaled.shape[1]
model = MRIFormer(
    n_features=n_features,
    d_model=D_MODEL,
    n_heads=N_HEADS,
    n_layers=N_LAYERS,
    d_ff=D_FF,
    dropout=DROPOUT,
    seq_len=SEQ_LEN,
    pred_len=PRED_LEN,
    resolutions=[1, 2, 4, 8]
).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

# Training
print("\nTraining MRIFormer...")
train_losses = []
val_losses = []

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    
    for batch_x, batch_y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        output = model(batch_x)
        loss = criterion(output.squeeze(), batch_y.squeeze())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        train_loss += loss.item()
    
    train_loss /= len(train_loader)
    train_losses.append(train_loss)
    
    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            output = model(batch_x)
            loss = criterion(output.squeeze(), batch_y.squeeze())
            val_loss += loss.item()
    
    val_loss /= len(test_loader)
    val_losses.append(val_loss)
    
    scheduler.step(val_loss)
    
    print(f"Epoch {epoch+1}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")

# Testing and evaluation
print("\nEvaluating on test set...")
model.eval()
predictions = []
actuals = []

with torch.no_grad():
    for batch_x, batch_y in test_loader:
        batch_x = batch_x.to(device)
        output = model(batch_x)
        predictions.extend(output.cpu().numpy().flatten())
        actuals.extend(batch_y.numpy().flatten())

predictions = np.array(predictions)
actuals = np.array(actuals)

# Inverse transform to original scale
consumption_idx = 2  # consumption column index
consumption_mean = scaler.mean_[consumption_idx]
consumption_std = scaler.scale_[consumption_idx]

predictions_original = predictions * consumption_std + consumption_mean
actuals_original = actuals * consumption_std + consumption_mean

# Calculate metrics
mae = np.mean(np.abs(predictions_original - actuals_original))
mse = np.mean((predictions_original - actuals_original) ** 2)
rmse = np.sqrt(mse)

print("\n" + "="*50)
print("FINAL RESULTS")
print("="*50)
print(f"MAE (Mean Absolute Error): {mae:.4f}")
print(f"MSE (Mean Squared Error): {mse:.4f}")
print(f"RMSE (Root Mean Squared Error): {rmse:.4f}")
print("="*50)

# Visualizations
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Training curves
axes[0, 0].plot(train_losses, label='Train Loss')
axes[0, 0].plot(val_losses, label='Validation Loss')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].set_title('Training and Validation Loss')
axes[0, 0].legend()
axes[0, 0].grid(True)

# Predictions vs Actuals
axes[0, 1].scatter(actuals_original, predictions_original, alpha=0.5, s=1)
axes[0, 1].plot([actuals_original.min(), actuals_original.max()], 
                [actuals_original.min(), actuals_original.max()], 'r--', lw=2)
axes[0, 1].set_xlabel('Actual Consumption')
axes[0, 1].set_ylabel('Predicted Consumption')
axes[0, 1].set_title('Predictions vs Actuals')
axes[0, 1].grid(True)

# Time series comparison (first 500 points)
plot_len = min(500, len(predictions_original))
axes[1, 0].plot(actuals_original[:plot_len], label='Actual', alpha=0.7)
axes[1, 0].plot(predictions_original[:plot_len], label='Predicted', alpha=0.7)
axes[1, 0].set_xlabel('Time Step')
axes[1, 0].set_ylabel('Consumption')
axes[1, 0].set_title(f'Time Series Comparison (First {plot_len} Points)')
axes[1, 0].legend()
axes[1, 0].grid(True)

# Error distribution
errors = predictions_original - actuals_original
axes[1, 1].hist(errors, bins=50, edgecolor='black', alpha=0.7)
axes[1, 1].axvline(x=0, color='r', linestyle='--', linewidth=2)
axes[1, 1].set_xlabel('Prediction Error')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].set_title('Error Distribution')
axes[1, 1].grid(True)

plt.tight_layout()
plt.savefig('mriformer_results.png', dpi=300, bbox_inches='tight')
print("\nResults saved to 'mriformer_results.png'")
plt.show()

# Save model
torch.save(model.state_dict(), 'mriformer_model.pth')
print("Model saved to 'mriformer_model.pth'")