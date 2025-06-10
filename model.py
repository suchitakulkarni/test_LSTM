# =============================================================================
# File: model.py
# =============================================================================

"""
Model definitions and related functions.
"""

# Import from common_imports
from common_imports import *
from config import config

class LSTMModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = None,
                 num_layers: int = None, dropout: float = None,
                 output_size: int = 1, bidirectional: bool = False):
        super(LSTMModel, self).__init__()
        
        # Use config defaults if not specified
        self.hidden_size = hidden_size or config.HIDDEN_SIZE
        self.num_layers = num_layers or config.NUM_LAYERS
        self.dropout = dropout or config.DROPOUT
        self.bidirectional = bidirectional
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout if self.num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        lstm_output_size = self.hidden_size * 2 if bidirectional else self.hidden_size
        self.fc = nn.Linear(lstm_output_size, output_size)
        self.dropout_layer = nn.Dropout(self.dropout)
        
    def forward(self, x):
        batch_size = x.size(0)
        num_directions = 2 if self.bidirectional else 1
        
        h0 = torch.zeros(self.num_layers * num_directions, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * num_directions, batch_size, self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.dropout_layer(out)
        out = self.fc(out)
        
        return out

def create_model(input_size: int, **kwargs) -> LSTMModel:
    """Factory function to create models with logging."""
    model = LSTMModel(input_size, **kwargs)
    logging.info(f"Created model with {sum(p.numel() for p in model.parameters())} parameters")
    return model.to(DEVICE)
