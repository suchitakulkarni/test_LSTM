# =============================================================================
# File: tuning.py
# =============================================================================

"""
Hyperparameter tuning using Optuna.
"""

# Import from common_imports and other modules
from common_imports import *
from config import config
from model import LSTMModel

class OptunaTrainer:
    def __init__(self, X_train, y_train, X_val, y_val):
        self.X_train = torch.FloatTensor(X_train)
        self.y_train = torch.FloatTensor(y_train)
        self.X_val = torch.FloatTensor(X_val)
        self.y_val = torch.FloatTensor(y_val)
        
    def objective(self, trial):
        # Suggest hyperparameters
        params = {
            'hidden_size': trial.suggest_categorical('hidden_size', [32, 64, 128, 256]),
            'num_layers': trial.suggest_int('num_layers', 1, 4),
            'dropout': trial.suggest_float('dropout', 0.0, 0.5),
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
            'bidirectional': trial.suggest_categorical('bidirectional', [True, False]),
        }
        
        # Create model
        model = LSTMModel(
            input_size=self.X_train.shape[-1],
            hidden_size=params['hidden_size'],
            num_layers=params['num_layers'],
            dropout=params['dropout'],
            bidirectional=params['bidirectional']
        ).to(DEVICE)
        
        # Create optimizer
        optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
        criterion = nn.MSELoss()
        
        # Create data loaders
        train_dataset = TensorDataset(self.X_train, self.y_train)
        val_dataset = TensorDataset(self.X_val, self.y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False)
        
        # Training loop with early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(config.EPOCHS):
            # Training
            model.train()
            train_loss = 0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs.squeeze(), batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
                    outputs = model(batch_x)
                    loss = criterion(outputs.squeeze(), batch_y)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= config.PATIENCE:
                    break
            
            # Report to Optuna
            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        
        return best_val_loss
    
    def optimize(self, n_trials: int = None):
        """Run hyperparameter optimization."""
        study = optuna.create_study(
            direction='minimize',
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10),
            sampler=TPESampler(seed=config.RANDOM_SEED)
        )
        
        n_trials = n_trials or config.N_TRIALS
        study.optimize(self.objective, n_trials=n_trials, timeout=config.OPTUNA_TIMEOUT)
        
        return study
