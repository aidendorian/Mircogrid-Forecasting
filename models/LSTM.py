import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

train = pd.read_csv('dataset/train.csv')
test = pd.read_csv('dataset/test.csv')

print("Data loaded successfully!")
print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
print(f"Columns: {train.columns.tolist()}")

train['lag_1'] = train['consumption'].shift(1)
train['lag_2'] = train['consumption'].shift(2)
train['rolling_mean'] = train['consumption'].rolling(window=5).mean()

test['lag_1'] = test['consumption'].shift(1)
test['lag_2'] = test['consumption'].shift(2)
test['rolling_mean'] = test['consumption'].rolling(window=5).mean()

selected_features = ['consumption', 'lag_1', 'lag_2', 'rolling_mean']
train_clean = train[selected_features].dropna()
test_clean = test[selected_features].dropna()

print(f"Original train shape: {train.shape}")
print(f"Clean train shape: {train_clean.shape}")
print(f"Original test shape: {test.shape}")
print(f"Clean test shape: {test_clean.shape}")

scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_clean)
test_scaled = scaler.transform(test_clean)

def create_sequences(data, seq_length=10):
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i, 1:])  # Features (exclude target from input)
        y.append(data[i, 0])  # Target (consumption)
    return np.array(X), np.array(y)

seq_length = 10
X_train, y_train = create_sequences(train_scaled, seq_length)
X_test, y_test = create_sequences(test_scaled, seq_length)

# Convert to tensors
X_train = torch.FloatTensor(X_train)
y_train = torch.FloatTensor(y_train).unsqueeze(1)
X_test = torch.FloatTensor(X_test)
y_test = torch.FloatTensor(y_test).unsqueeze(1)

print(f"Training sequences shape: {X_train.shape}")
print(f"Training targets shape: {y_train.shape}")

train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 6. Define model
class LSTMForecast(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=0.2):
        super(LSTMForecast, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

        # Initialize weights
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, x):
        batch_size = x.size(0)
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device="cuda")
        c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device="cuda")
        lstm_out, _ = self.lstm(x, (h_0, c_0))
        out = self.fc(lstm_out[:, -1, :])  # Take last output
        return out

# 7. Initialize model, loss, optimizer
model = LSTMForecast(input_size=3, hidden_size=64, output_size=1, num_layers=2, dropout=0.2)
model.to("cuda")
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

# 8. Training function
def train_model(model, train_loader, criterion, optimizer, scheduler, num_epochs=20):
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
            x_batch = x_batch.to("cuda")
            y_batch = y_batch.to("cuda")
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()

            # Gradient clipping to prevent exploding gradients
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()

            epoch_loss += loss.item()

        scheduler.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(train_loader):.4f}')

# 9. Train the model
print("Training the model...")
train_model(model, train_loader, criterion, optimizer, scheduler, num_epochs=20)

# 10. Make predictions
model.eval()
with torch.no_grad():
    y_pred = model(X_test.to("cuda")).to("cpu").squeeze().numpy()

y_true = y_test.squeeze().numpy()

# 11. Calculate metrics
def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_pred - y_true) ** 2))

def mse(y_true, y_pred):
    return np.mean((y_pred - y_true) ** 2)

def mae(y_true, y_pred):
    return np.mean(np.abs(y_pred - y_true))

def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

print(f"\nFinal Results:")
print(f"MSE: {mse(y_true, y_pred):.4f}")
print(f"RMSE: {rmse(y_true, y_pred):.4f}")
print(f"MAE: {mae(y_true, y_pred):.4f}")
print(f"MAPE: {mape(y_true, y_pred):.2f}%")