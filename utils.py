# =============================================================================
# File: utils.py
# =============================================================================

"""
Utility functions for the ML project.
"""

from common_imports import *
from config import config

def save_model(model, filepath: str, metadata: dict = None):
    """Save model with metadata."""
    save_dict = {
        'model_state_dict': model.state_dict(),
        'model_class': model.__class__.__name__,
        'timestamp': datetime.now().isoformat(),
        'metadata': metadata or {}
    }
    torch.save(save_dict, filepath)
    logging.info(f"Model saved to {filepath}")

def load_model(filepath: str, model_class, **model_kwargs):
    """Load model from file."""
    checkpoint = torch.load(filepath, map_location=DEVICE)
    model = model_class(**model_kwargs)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    logging.info(f"Model loaded from {filepath}")
    return model, checkpoint.get('metadata', {})

def plot_training_history(train_losses: List[float], val_losses: List[float],
                         save_path: str = None):
    """Plot training history."""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', alpha=0.7)
    plt.plot(val_losses, label='Validation Loss', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"Plot saved to {save_path}")
    
    plt.show()

def create_sequences(data: np.ndarray, seq_length: int, target_col: int = -1):
    """Create sequences for time series data."""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length, target_col])
    return np.array(X), np.array(y)
