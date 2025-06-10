from common_imports import *
from config import config

# --------------------
# 2. Sliding Window
# --------------------

sfeatures = ['s' + str(i) for i in range(1, 22)]
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

def read_data(traindatafile = './CMAPSSData/train_FD001.txt', \
              testdatafile = './CMAPSSData/test_FD001.txt', \
              labelfile = './CMAPSSData/RUL_FD001.txt'):
    col_names = ['unit_no', 'time', 'op1', 'op2', 'op3'] + sfeatures

    data_train = pd.read_csv(datafile, sep=r'\s+', header=None, names=col_names)
    data_test = pd.read_csv(datafile, sep=r'\s+', header=None, names=col_names)
    RUL_labels = pd.read_csv(datafile, sep=r'\s+', header=None, names=['RUL'])
    return [data_train, data_test, RUL_labels]

def create_traintest_dataframe(data_train, data_test, RUL_labels):
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
    return data_train, data_test

def scale_and_split(data_train, data_test):
    feature_cols = ['op1', 'op2', 'op3'] + sfeatures
    # Normalize
    scaler = StandardScaler()
    data_train[feature_cols] = scaler.fit_transform(data_train[feature_cols])
    data_test[feature_cols] = scaler.transform(data_test[feature_cols])
    train_ids, valid_ids = train_test_split(data_train['unit_no'].unique(), test_size=0.2, random_state=42)
    train_df = data_train[data_train['unit_no'].isin(train_ids)]
    valid_df = data_train[data_train['unit_no'].isin(valid_ids)]

    # 1. Fix feature selection (remove data leakage)
    feature_cols = ['op1', 'op2', 'op3'] + sfeatures
    # DO NOT include RUL_calc in features - that's data leakage!
    # Only include unit_no for grouping, but exclude from model input
    data_features_train = data_train[feature_cols + ['unit_no', 'RUL_calc']]
    data_features_test = data_test[feature_cols + ['unit_no', 'RUL_calc']]

    x_train, y_train, uid_train = create_sliding_window(train_df)
    x_valid, y_valid, uid_valid = create_sliding_window(valid_df)
    x_test, y_test, uid_test = create_sliding_window(data_test)

    # Flatten
    X_train = x_train.reshape(x_train.shape[0], -1)
    X_valid = x_valid.reshape(x_valid.shape[0], -1)
    X_test = x_test.reshape(x_test.shape[0], -1)
    return X_train, X_valid, X_test

