#!/usr/bin/env python3
"""
Corrected training script with proper QuantileLoss implementation
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

# Custom QuantileLoss that handles the correct tensor dimensions
class CorrectedQuantileLoss(nn.Module):
    """Corrected Quantile Loss that properly handles tensor dimensions"""
    
    def __init__(self, quantiles):
        super().__init__()
        self.quantiles = quantiles
        
    def forward(self, preds, target):
        """
        Args:
            preds: (batch_size, seq_len, num_quantiles) 
            target: (batch_size, seq_len)
        """
        assert not target.requires_grad
        assert preds.size(0) == target.size(0)
        assert preds.size(1) == target.size(1)
        
        losses = []
        for i, q in enumerate(self.quantiles):
            # Extract predictions for this quantile: (batch_size, seq_len)
            pred_q = preds[:, :, i]
            
            # Calculate errors: (batch_size, seq_len)
            errors = target - pred_q
            
            # Apply quantile loss formula
            loss_q = torch.max((q - 1) * errors, q * errors)
            losses.append(loss_q)
        
        # Stack losses and compute mean
        losses = torch.stack(losses, dim=-1)  # (batch_size, seq_len, num_quantiles)
        loss = torch.mean(losses)
        
        return loss


# Import the TFT model components
try:
    from models.TFTransformer import TFT
except ImportError:
    try:
        from models.TFTransformer import TFT
    except ImportError:
        print("ERROR: Could not import TFT model components.")
        sys.exit(1)


class EnergyDataset(Dataset):
    """Fixed PyTorch Dataset for energy consumption prediction"""
    
    def __init__(self, data, config, feature_columns, mode='train'):
        self.data = data.reset_index(drop=True)
        self.config = config
        self.feature_columns = feature_columns
        self.mode = mode
        self.seq_length = config['seq_length']
        self.encode_length = config['encode_length'] 
        self.predict_length = self.seq_length - self.encode_length
        
        print(f"Dataset config: seq_length={self.seq_length}, encode_length={self.encode_length}, predict_length={self.predict_length}")
        
        # Create sequences
        self.sequences = self._create_sequences()
        
    def _create_sequences(self):
        """Create sequential data samples"""
        sequences = []
        
        if len(self.data) < self.seq_length:
            print(f"Warning: Data length {len(self.data)} is less than sequence length {self.seq_length}")
            return sequences
        
        for i in range(len(self.data) - self.seq_length + 1):
            seq_data = self.data.iloc[i:i + self.seq_length].copy()
            sequences.append(seq_data)
            
        print(f"Created {len(sequences)} sequences for {self.mode}")
        return sequences
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        if idx >= len(self.sequences):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.sequences)}")
            
        seq = self.sequences[idx]
        
        # Create identifier (static features) - shape: (1, static_variables)
        identifier = torch.zeros(1, self.config['static_variables'], dtype=torch.long)
        
        # Prepare input features - shape: (seq_length, num_features)
        input_features = seq[self.feature_columns].values.astype(np.float32)
        inputs = torch.tensor(input_features, dtype=torch.float32)
        
        # Extract target values from the decoder part only
        target_values = seq[self.config['target_column']].values[self.encode_length:].astype(np.float32)
        targets = torch.tensor(target_values, dtype=torch.float32)
        
        return {
            'identifier': identifier,
            'inputs': inputs,
            'targets': targets
        }


class DataPreprocessor:
    """Enhanced data preprocessing for TFT model"""
    
    def __init__(self, config):
        self.config = config
        self.scalers = {}
        self.encoders = {}
        self.feature_columns = []
        self.numeric_features = []
        self.categorical_features = []
        
    def _extract_time_features(self, df):
        """Extract time-based features from date column"""
        df = df.copy()
        df[self.config['date_column']] = pd.to_datetime(df[self.config['date_column']])
        
        # Extract time features
        df['hour'] = df[self.config['date_column']].dt.hour
        df['day_of_week'] = df[self.config['date_column']].dt.dayofweek
        df['month'] = df[self.config['date_column']].dt.month - 1  # 0-indexed for embedding
        
        return df
    
    def _identify_feature_types(self, df):
        """Identify numeric and categorical features"""
        # Time-based categorical features
        self.categorical_features = ['hour', 'day_of_week', 'month']
        
        # All other features (except date and target) are numeric
        exclude_cols = [self.config['date_column'], self.config['target_column']] + self.categorical_features
        self.numeric_features = [col for col in df.columns if col not in exclude_cols]
        
        # Combined feature list (order matters for the model)
        self.feature_columns = self.numeric_features + self.categorical_features
        
        print(f"Feature identification:")
        print(f"  - Numeric features ({len(self.numeric_features)}): {self.numeric_features[:5]}{'...' if len(self.numeric_features) > 5 else ''}")
        print(f"  - Categorical features ({len(self.categorical_features)}): {self.categorical_features}")
        
    def fit_transform(self, df):
        """Fit preprocessors and transform training data"""
        print("Fitting data preprocessors...")
        df = df.copy()
        
        # Sort by date
        df = df.sort_values(self.config['date_column']).reset_index(drop=True)
        
        # Extract time features
        df = self._extract_time_features(df)
        
        # Identify feature types
        self._identify_feature_types(df)
        
        # Scale numeric features
        print(f"Scaling {len(self.numeric_features)} numeric features...")
        for col in self.numeric_features:
            if col in df.columns:
                scaler = StandardScaler()
                df[col] = scaler.fit_transform(df[col].values.reshape(-1, 1)).flatten()
                self.scalers[col] = scaler
        
        # Scale target
        target_scaler = StandardScaler()
        df[self.config['target_column']] = target_scaler.fit_transform(
            df[self.config['target_column']].values.reshape(-1, 1)
        ).flatten()
        self.scalers[self.config['target_column']] = target_scaler
        
        # Ensure categorical features are in correct range
        df['hour'] = df['hour'].clip(0, 23)
        df['day_of_week'] = df['day_of_week'].clip(0, 6) 
        df['month'] = df['month'].clip(0, 11)
        
        return df
    
    def transform(self, df):
        """Transform validation/test data using fitted preprocessors"""
        df = df.copy()
        
        # Sort by date
        df = df.sort_values(self.config['date_column']).reset_index(drop=True)
        
        # Extract time features
        df = self._extract_time_features(df)
        
        # Apply fitted scalers
        for col in self.numeric_features:
            if col in df.columns and col in self.scalers:
                df[col] = self.scalers[col].transform(df[col].values.reshape(-1, 1)).flatten()
        
        # Scale target
        if self.config['target_column'] in df.columns:
            df[self.config['target_column']] = self.scalers[self.config['target_column']].transform(
                df[self.config['target_column']].values.reshape(-1, 1)
            ).flatten()
        
        # Ensure categorical features are in correct range
        df['hour'] = df['hour'].clip(0, 23)
        df['day_of_week'] = df['day_of_week'].clip(0, 6)
        df['month'] = df['month'].clip(0, 11)
        
        return df
    
    def inverse_transform_target(self, values):
        """Inverse transform target values to original scale"""
        if isinstance(values, torch.Tensor):
            values = values.cpu().numpy()
        
        if len(values.shape) == 1:
            values = values.reshape(-1, 1)
        elif len(values.shape) == 2 and values.shape[0] == 1:
            values = values.T
        
        return self.scalers[self.config['target_column']].inverse_transform(values).flatten()


def create_config_dict(train_df, preprocessor):
    """Create model configuration dictionary"""
    
    # Count feature types
    categorical_cols = preprocessor.categorical_features
    numeric_cols = preprocessor.numeric_features
    
    config = {
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'batch_size': 64,
        'seq_length': 168,        # 7 days * 24 hours
        'encode_length': 144,     # 5 days * 24 hours  
        'predict_length': 24,     # Will be calculated as seq_length - encode_length
        
        # Model architecture
        'lstm_hidden_dimension': 128,
        'lstm_layers': 2,
        'embedding_dim': 16,
        'attn_heads': 4,
        'dropout': 0.1,
        
        # Features
        'static_variables': 1,  # Dummy static variable
        'time_varying_categorical_variables': len(categorical_cols),
        'time_varying_real_variables_encoder': len(numeric_cols),
        'time_varying_real_variables_decoder': len(numeric_cols),
        'num_masked_series': 0,
        
        # Quantile prediction
        'num_quantiles': 3,
        'vailid_quantiles': [0.1, 0.5, 0.9],
        
        # Embedding vocab sizes
        'static_embedding_vocab_sizes': [1],  # Dummy
        'time_varying_embedding_vocab_sizes': [24, 7, 12],  # hour, day_of_week, month
        
        # Data columns
        'target_column': 'consumption',
        'date_column': 'date',
    }
    
    print(f"Model configuration:")
    print(f"  - Sequence length: {config['seq_length']}")
    print(f"  - Encode length: {config['encode_length']}")  
    print(f"  - Prediction length: {config['seq_length'] - config['encode_length']}")
    print(f"  - Numeric features: {len(numeric_cols)}")
    print(f"  - Categorical features: {len(categorical_cols)}")
    print(f"  - Quantiles: {config['num_quantiles']}")
    
    return config


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch with corrected loss calculation"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, batch in enumerate(train_loader):
        try:
            optimizer.zero_grad()
            
            # Prepare batch data
            inputs = {
                'identifier': batch['identifier'].to(device),
                'inputs': batch['inputs'].to(device)
            }
            targets = batch['targets'].to(device)  # Shape: (batch_size, predict_length)
            
            # Forward pass
            outputs, *_ = model(inputs)
            # outputs shape: (batch_size, predict_length, num_quantiles)
            
            # Calculate loss using corrected QuantileLoss
            loss = criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 50 == 0:
                print(f'  Batch {batch_idx:3d}/{len(train_loader)}, Loss: {loss.item():.6f}')
                
        except Exception as e:
            print(f"Error in batch {batch_idx}: {e}")
            print(f"  Output shape: {outputs.shape if 'outputs' in locals() else 'Not computed'}")
            print(f"  Target shape: {targets.shape}")
            import traceback
            traceback.print_exc()
            continue
    
    return total_loss / max(num_batches, 1)


def validate_epoch(model, val_loader, criterion, device):
    """Validate for one epoch with corrected loss calculation"""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in val_loader:
            try:
                inputs = {
                    'identifier': batch['identifier'].to(device),
                    'inputs': batch['inputs'].to(device)
                }
                targets = batch['targets'].to(device)
                
                outputs, *_ = model(inputs)
                loss = criterion(outputs, targets)
                
                total_loss += loss.item()
                num_batches += 1
                
            except Exception as e:
                print(f"Validation error: {e}")
                continue
    
    return total_loss / max(num_batches, 1)


def train_model(model, train_loader, val_loader, config):
    """Main training loop with corrected loss function"""
    print("Starting model training...")
    
    device = config['device']
    model.to(device)
    
    # Use corrected loss function
    criterion = CorrectedQuantileLoss(config['vailid_quantiles'])
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.02)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    # Training tracking
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = 20
    
    for epoch in range(10):  # Max epochs
        print(f"\nEpoch {epoch+1}/100")
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        
        # Validate
        val_loss = validate_epoch(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print progress
        print(f'Epoch {epoch+1:3d}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, LR: {current_lr:.8f}')
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            Path('models').mkdir(exist_ok=True)
            torch.save(model.state_dict(), 'models/best_tft_model.pth')
            print(f'  → New best model saved (val_loss: {val_loss:.6f})')
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    return train_losses, val_losses


def evaluate_model(model, test_loader, preprocessor, config):
    """Evaluate model and calculate metrics"""
    print("Evaluating model...")
    
    device = config['device']
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            try:
                inputs = {
                    'identifier': batch['identifier'].to(device),
                    'inputs': batch['inputs'].to(device)
                }
                targets = batch['targets'].cpu().numpy()
                
                outputs, *_ = model(inputs)
                
                # Use median quantile for evaluation
                median_idx = len(config['vailid_quantiles']) // 2
                predictions = outputs[:, :, median_idx].cpu().numpy()
                
                all_predictions.append(predictions)
                all_targets.append(targets)
                
            except Exception as e:
                print(f"Evaluation error: {e}")
                continue
    
    if not all_predictions:
        print("No valid predictions generated!")
        return None, None, None, None, None
    
    # Concatenate all results
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    # Inverse transform to original scale
    predictions_orig = []
    targets_orig = []
    
    for i in range(len(all_predictions)):
        pred_orig = preprocessor.inverse_transform_target(all_predictions[i])
        target_orig = preprocessor.inverse_transform_target(all_targets[i])
        predictions_orig.append(pred_orig)
        targets_orig.append(target_orig)
    
    predictions_orig = np.array(predictions_orig)
    targets_orig = np.array(targets_orig)
    
    # Calculate metrics
    mae = mean_absolute_error(targets_orig.flatten(), predictions_orig.flatten())
    mse = mean_squared_error(targets_orig.flatten(), predictions_orig.flatten())
    rmse = np.sqrt(mse)
    
    print(f"\nTest Metrics:")
    print(f"MAE:  {mae:.4f}")
    print(f"MSE:  {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    
    return mae, mse, rmse, predictions_orig, targets_orig


def create_data_loaders(train_df, val_df, test_df, config, preprocessor):
    """Create PyTorch data loaders"""
    print("Creating data loaders...")
    
    # Create datasets
    train_dataset = EnergyDataset(train_df, config, preprocessor.feature_columns, mode='train')
    val_dataset = EnergyDataset(val_df, config, preprocessor.feature_columns, mode='val')
    test_dataset = EnergyDataset(test_df, config, preprocessor.feature_columns, mode='test')
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        drop_last=True,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        drop_last=True,
        num_workers=0
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        drop_last=True,
        num_workers=0
    )
    
    print(f"Data loader info:")
    print(f"  - Train batches: {len(train_loader)}")
    print(f"  - Validation batches: {len(val_loader)}")
    print(f"  - Test batches: {len(test_loader)}")
    
    return train_loader, val_loader, test_loader


def plot_results(train_losses, val_losses, predictions, targets, save_path='results'):
    """Plot training curves and sample predictions"""
    Path(save_path).mkdir(exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Training curves
    axes[0, 0].plot(train_losses, label='Train Loss', alpha=0.8)
    axes[0, 0].plot(val_losses, label='Validation Loss', alpha=0.8)
    axes[0, 0].set_title('Training Curves')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Sample predictions
    num_samples = min(3, len(predictions))
    for i in range(num_samples):
        if i == 0:
            ax = axes[0, 1]
        elif i == 1:
            ax = axes[1, 0]
        else:
            ax = axes[1, 1]
        
        time_steps = range(len(targets[i]))
        ax.plot(time_steps, targets[i], label='Actual', marker='o', alpha=0.8, markersize=3)
        ax.plot(time_steps, predictions[i], label='Predicted', marker='s', alpha=0.8, markersize=3)
        ax.set_title(f'Sample Prediction {i+1}')
        ax.set_xlabel('Time Steps (Hours)')
        ax.set_ylabel('Consumption')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/training_results.png', dpi=300, bbox_inches='tight')
    print(f"Results plot saved to {save_path}/training_results.png")
    plt.show()


def main():
    """Main training pipeline with corrected loss function"""
    print("=" * 60)
    print("TFT Training with Corrected QuantileLoss")
    print("=" * 60)
    
    # Create directories
    Path('models').mkdir(exist_ok=True)
    Path('results').mkdir(exist_ok=True)
    
    # Load data
    print("\nLoading data...")
    try:
        train_df = pd.read_csv('dataset/train.csv')
        test_df = pd.read_csv('dataset/test.csv')
        print(f"Train data: {train_df.shape}")
        print(f"Test data:  {test_df.shape}")
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        return
    
    # Initialize preprocessor
    basic_config = {
        'target_column': 'consumption',
        'date_column': 'date'
    }
    
    preprocessor = DataPreprocessor(basic_config)
    
    # Preprocess data
    print("\nPreprocessing data...")
    train_processed = preprocessor.fit_transform(train_df)
    test_processed = preprocessor.transform(test_df)
    
    # Create full config based on processed data
    config = create_config_dict(train_processed, preprocessor)
    
    # Create validation split
    val_size = int(0.2 * len(train_processed))
    val_processed = train_processed[-val_size:].reset_index(drop=True)
    train_processed = train_processed[:-val_size].reset_index(drop=True)
    
    print(f"\nData splits:")
    print(f"  - Train:      {len(train_processed):,} samples")
    print(f"  - Validation: {len(val_processed):,} samples")
    print(f"  - Test:       {len(test_processed):,} samples")
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        train_processed, val_processed, test_processed, config, preprocessor
    )
    
    # Create model
    print("\nCreating model...")
    try:
        model = TFT(config)
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model created successfully!")
        print(f"Total trainable parameters: {total_params:,}")
    except Exception as e:
        print(f"Error creating model: {e}")
        return
    
    # Test model with a single batch
    print("\nTesting model with sample batch...")
    try:
        sample_batch = next(iter(train_loader))
        inputs = {
            'identifier': sample_batch['identifier'].to(config['device']),
            'inputs': sample_batch['inputs'].to(config['device'])
        }
        targets = sample_batch['targets']
        
        model.to(config['device'])
        with torch.no_grad():
            outputs, *_ = model(inputs)
        
        print(f"✓ Model test successful!")
        print(f"  Input shape: {inputs['inputs'].shape}")
        print(f"  Output shape: {outputs.shape}")
        print(f"  Target shape: {targets.shape}")
        
        # Test corrected loss function
        criterion = CorrectedQuantileLoss(config['vailid_quantiles'])
        test_loss = criterion(outputs, targets.to(config['device']))
        print(f"  Test loss: {test_loss.item():.6f}")
        
    except Exception as e:
        print(f"✗ Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Train model
    print("\n" + "="*40)
    print("STARTING TRAINING")
    print("="*40)
    
    train_losses, val_losses = train_model(model, train_loader, val_loader, config)
    
    # Load best model for evaluation
    print(f"\nLoading best model for evaluation...")
    model.load_state_dict(torch.load('models/best_tft_model.pth', map_location=config['device']))
    
    # Evaluate model
    mae, mse, rmse, predictions, targets = evaluate_model(model, test_loader, preprocessor, config)
    
    if predictions is not None:
        # Plot results
        plot_results(train_losses, val_losses, predictions, targets)
        
        # Save results
        results = {
            'config': config,
            'metrics': {'mae': mae, 'mse': mse, 'rmse': rmse},
            'train_losses': train_losses,
            'val_losses': val_losses
        }
        
        torch.save(results, 'results/training_results.pth')
        print(f"Results saved to results/training_results.pth")
    
    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()