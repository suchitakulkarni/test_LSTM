# =============================================================================
# File: common_imports.py
# =============================================================================

"""
Common imports for the ML project.
This file centralizes all commonly used imports to avoid repetition.
"""

# Standard library imports
import os
import sys
import json
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Data manipulation and analysis
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score,
    classification_report, confusion_matrix
)

# Deep learning frameworks
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset

# Optional: TensorFlow/Keras (comment out if not needed)
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers

# Hyperparameter tuning
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Optional: Additional visualization libraries
# import plotly.express as px
# import plotly.graph_objects as go

# Progress bars and utilities
from tqdm import tqdm
import time

# Configuration and logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Set random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")
