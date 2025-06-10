# =============================================================================
# File: config.py
# =============================================================================

"""
Configuration file for the ML project.
Contains all hyperparameters, paths, and settings.
"""

from dataclasses import dataclass
from pathlib import Path

@dataclass
class Config:
    # Project paths
    PROJECT_ROOT: Path = Path(__file__).parent
    DATA_DIR: Path = PROJECT_ROOT / "data"
    MODELS_DIR: Path = PROJECT_ROOT / "models"
    LOGS_DIR: Path = PROJECT_ROOT / "logs"
    RESULTS_DIR: Path = PROJECT_ROOT / "results"
    
    # Data parameters
    TRAIN_SIZE: float = 0.7
    VAL_SIZE: float = 0.15
    TEST_SIZE: float = 0.15
    BATCH_SIZE: int = 32
    
    # Model parameters
    HIDDEN_SIZE: int = 128
    NUM_LAYERS: int = 2
    DROPOUT: float = 0.2
    LEARNING_RATE: float = 0.001
    
    # Training parameters
    EPOCHS: int = 100
    PATIENCE: int = 10
    
    # Optuna parameters
    N_TRIALS: int = 50
    OPTUNA_TIMEOUT: int = 3600  # 1 hour
    
    # Other settings
    RANDOM_SEED: int = 42
    VERBOSE: bool = True
    
    def __post_init__(self):
        # Create directories if they don't exist
        for dir_path in [self.DATA_DIR, self.MODELS_DIR, self.LOGS_DIR, self.RESULTS_DIR]:
            dir_path.mkdir(exist_ok=True)

# Global config instance
config = Config()
