# =============================================================================
# File: main.py
# =============================================================================

"""
Main execution script that ties everything together.
"""

# Import everything from common imports and project modules
from common_imports import *
from config import config
from model import create_model, LSTMModel
from tuning import OptunaTrainer
from utils import save_model, plot_training_history, create_sequences
import sys
##from  dataprocessing import x_train, y_train, uid_train, x_valid, y_valid, uid_valid, x_test, y_test, uid_test
#from dataprocessing import  X_train, X_valid, X_test

def main():
    logging.info("Starting ML pipeline...")
    # --------------------
    # 1. Load and Preprocess
    # --------------------
    sfeatures = ['s' + str(i) for i in range(1, 22)]
    col_names = ['unit_no', 'time', 'op1', 'op2', 'op3'] + sfeatures

    data_train = pd.read_csv('./CMAPSSData/train_FD001.txt', sep=r'\s+', header=None, names=col_names)
    data_test = pd.read_csv('./CMAPSSData/test_FD001.txt', sep=r'\s+', header=None, names=col_names)
    RUL_labels = pd.read_csv('./CMAPSSData/RUL_FD001.txt', sep=r'\s+', header=None, names=['RUL'])

    # Add RUL to train
    max_times = data_train.groupby('unit_no')['time'].max().reset_index()
    data_train = data_train.merge(max_times, on='unit_no', suffixes=('', '_max'))
    data_train['RUL_calc'] = data_train['time_max'] - data_train['time']
    data_train.drop(['time', 'time_max'], axis=1, inplace=True)

    # Add RUL to test
    RUL_labels['unit_no'] = RUL_labels.index + 1
    max_times_test = data_test.groupby('unit_no')['time'].max().reset_index()
    data_test = data_test.merge(max_times_test, on='unit_no', suffixes=('', '_max'))
    data_test = data_test.merge(RUL_labels, on='unit_no')
    data_test['RUL_calc'] = data_test['time_max'] - data_test['time'] + data_test['RUL']
    data_test.drop(['time', 'time_max', 'RUL'], axis=1, inplace=True)

    feature_cols = ['op1', 'op2', 'op3'] + sfeatures

    # Normalize
    scaler = StandardScaler()
    data_train[feature_cols] = scaler.fit_transform(data_train[feature_cols])
    data_test[feature_cols] = scaler.transform(data_test[feature_cols])
    data_train.head()

    # --------------------
    # 3. Proper Train/Validation Split by unit_no
    # --------------------
    train_ids, valid_ids = train_test_split(data_train['unit_no'].unique(), test_size=0.2, random_state=42)
    train_df = data_train[data_train['unit_no'].isin(train_ids)]
    valid_df = data_train[data_train['unit_no'].isin(valid_ids)]

    # --------------------
    # 2. Sliding Window
    # --------------------
    def create_sliding_window(df, window_length=30):
        x, y, unit_id = [], [], []
        for unit in df['unit_no'].unique():
            unit_df = df[df['unit_no'] == unit]
            for i in range(len(unit_df) - window_length):
                window = unit_df.iloc[i:i + window_length][feature_cols].values
                label = unit_df.iloc[i + window_length]['RUL_calc']
                uid = unit_df.iloc[i + window_length]['unit_no']
                unit_id.append(uid)
                x.append(window)
                y.append(label)
        return np.array(x), np.array(y), np.asarray(unit_id)

    x_train, y_train, uid_train = create_sliding_window(train_df)
    x_valid, y_valid, uid_valid = create_sliding_window(valid_df)
    x_test, y_test, uid_test = create_sliding_window(data_test)

    # Hyperparameter tuning
    logging.info("Starting hyperparameter optimization...")
    trainer = OptunaTrainer(x_train, y_train, x_valid, y_valid)
    study = trainer.optimize(n_trials=20)  # Reduced for demo
    
    logging.info(f"Best parameters: {study.best_params}")
    logging.info(f"Best validation loss: {study.best_value:.6f}")
    
    # Train final model
    best_params = study.best_params
    final_model = create_model(
        input_size=X_train.shape[-1],
        hidden_size=best_params['hidden_size'],
        num_layers=best_params['num_layers'],
        dropout=best_params['dropout'],
        bidirectional=best_params['bidirectional']
    )
    
    # Save model
    save_model(
        final_model,
        config.MODELS_DIR / "best_model.pth",
        metadata={'best_params': best_params, 'study_value': study.best_value}
    )
    
    logging.info("Pipeline completed successfully!")

if __name__ == "__main__":
    main()
