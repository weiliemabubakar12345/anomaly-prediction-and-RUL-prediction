import pandas as pd
import sys
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, RepeatVector, TimeDistributed, Conv1D, MaxPooling1D, UpSampling1D, Flatten, Reshape, Lambda, Concatenate, BatchNormalization, Add
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score, classification_report
import seaborn as sns
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.layers import GRU, RepeatVector, TimeDistributed

# Define the column names
column_names = ['unit_number', 'time_in_cycles', 'setting_1', 'setting_2', 'setting_3',
                'T2[°R]', 'T24[°R]', 'T30[°R]', 'T50[°R]',
                'P2[psia]', 'P15[psia]', 'P30[psia]',
                'Nf[rpm]', 'Nc[rpm]', 'epr[--]',
                'Ps30[psia]', 'phi[pps/psi]', 'NRf[rpm]',
                'NRc[rpm]', 'BPR[--]', 'farB[--]',
                'htBleed[--]', 'Nf_dmd[rpm]',
                'PCNfR_dmd[rpm]', 'W31[lbm/s]', 'W32[lbm/s]']

# Define the base directory where your dataset is stored
base_directory = r'D:\MITLAB-Abu\Preventive Maintenance\1. Dataset'
# base_directory = '/Users/weiliemabubakar/Documents/Preventive Maintenance/1. Dataset' 
#or you can try to use the path to your dataset
# Updated for replication compliance: looks for data in the local folder
# base_directory = os.path.join(os.getcwd(), 'Dataset')

# List of the four training file names
file_names_training = ['train_FD001.txt', 'train_FD002.txt', 'train_FD003.txt', 'train_FD004.txt']

# List of the four testing file names
file_names_testing = ['test_FD001.txt', 'test_FD002.txt', 'test_FD003.txt', 'test_FD004.txt']

# List of the four RUL file names
file_names_rul = ['RUL_FD001.txt', 'RUL_FD002.txt', 'RUL_FD003.txt', 'RUL_FD004.txt']

# Dictionary to store the DataFrames
train_datasets = {}
test_datasets = {}
rul_datasets = {}

# Function to create sequences
def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length + 1):
        sequences.append(data[i:i+seq_length])
    return np.array(sequences)

sequence_length = 50  # Example sequence length

# Loop through each file and load it
for file in file_names_training:
    # Extract the dataset name (e.g., FD001)
    dataset_name = file.split('.')[0]  # Removes '.txt'
    
    # Read the file
    df = pd.read_csv(os.path.join(base_directory, file), sep='\s+', header=None, names=column_names)

    # Add a column to identify which dataset it came from (optional, for comparison)
    df['dataset'] = dataset_name

    # --- Step 1: Select Sensor Columns ---
    sensor_columns = [col for col in column_names if col not in ['unit_number', 'time_in_cycles', 'setting_1', 'setting_2', 'setting_3']]
    data_to_normalize = df[sensor_columns]
    
    
    
    #  # --- Step 2: Normalize the Data ---
    scaler = StandardScaler()
    
    # Identify the healthy phase (e.g., first 60% of each engine's life)
    df['cycle_norm'] = df.groupby('unit_number')['time_in_cycles'].transform(lambda x: x / x.max())
    healthy_mask = df['cycle_norm'] <= 0.6
    
    # Fit the scaler on healthy data only
    healthy_data = data_to_normalize[healthy_mask]
    scaler.fit(healthy_data)
    
    # Transform the entire dataset
    normalized_data = scaler.transform(data_to_normalize)
    
    #--Step 3: Create sequences--
    X_sequences = create_sequences(normalized_data, sequence_length)
    y_sequences = X_sequences
    
    #store the processed sequences and labels
    train_datasets[dataset_name] = {
        'X': X_sequences,
        'y': y_sequences,
        'normalized_data': normalized_data,
        'scaler': scaler,
        'df':df
    }
    
    print(f"Successfully processed {dataset_name}.")
    print(f"  Shape of sequences: {X_sequences.shape}")

for file in file_names_testing:
    dataset_name = file.split('.')[0]  # Removes '.txt'
    df = pd.read_csv(os.path.join(base_directory, file), sep='\s+', header=None, names=column_names)
    df['dataset'] = dataset_name
    
    sensor_columns = [col for col in column_names if col not in ['unit_number', 'time_in_cycles', 'setting_1', 'setting_2', 'setting_3']]
    data_to_normalize = df[sensor_columns]

    corresponding_train_dataset = dataset_name.replace('test', 'train')
    scaler = train_datasets[corresponding_train_dataset]['scaler']  # Use the scaler from the training dataset
    normalized_data = scaler.transform(data_to_normalize)

    # Create sequences
    X_sequences = create_sequences(normalized_data, sequence_length)

    # Store processed test data
    test_datasets[dataset_name] = {
        'X': X_sequences,
        'df': df,
        'normalized_data': normalized_data,
        'scaler': scaler
    }
    
    print(f"Successfully processed test dataset {dataset_name}. Sequence shape: {X_sequences.shape}")
    
for file in file_names_rul:
    dataset_name = file.split('.')[0]  # Removes '.txt'
    rul_df = pd.read_csv(os.path.join(base_directory, file), header=None, names=['RUL'])
    rul_datasets[dataset_name] = rul_df
    
    print(f"Successfully processed RUL dataset {dataset_name}. Shape: {rul_df.shape}")

def build_cnn_lstm_autoencoder(input_shape):
    #CNN
    # inputs = Input(shape=input_shape)
    # conv1 = Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(inputs)
    # pool1 = MaxPooling1D(pool_size=2, padding='same')(conv1)
    # conv2 = Conv1D(filters=16, kernel_size=3, activation='relu', padding='same')(pool1)
    # pool2 = MaxPooling1D(pool_size=2, padding='same')(conv2)
    
    inputs = Input(shape=input_shape)
    conv1 = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(inputs)
    pool1 = MaxPooling1D(pool_size=2, padding='same')(conv1)
    pool2 = MaxPooling1D(pool_size=2, padding='same')(pool1)

    encoded_lstm = pool2

    #encoder
    encoded = LSTM(96, activation='tanh', return_sequences=True)(encoded_lstm)
    encoded = Dropout(0.2)(encoded)
    encoded = LSTM(48, activation='tanh', return_sequences=True)(encoded)
    encoded = Dropout(0.2)(encoded)
    encoded = LSTM(24, activation='tanh', return_sequences=True)(encoded)
    encoded = Dropout(0.2)(encoded)
    encoded = LSTM(12, activation='tanh')(encoded)  # Smaller bottleneck
    
    #Bottleneck
    decoded = RepeatVector(input_shape[0])(encoded)
    
    #decoder
    decoded = LSTM(12, activation='tanh', return_sequences=True)(decoded)
    decoded = Dropout(0.2)(decoded)
    decoded = LSTM(24, activation='tanh', return_sequences=True)(decoded)
    decoded = Dropout(0.2)(decoded)
    decoded = LSTM(48, activation='tanh', return_sequences=True)(decoded)
    decoded = Dropout(0.2)(decoded)
    decoded = LSTM(96, activation='tanh', return_sequences=True)(decoded)
    outputs = TimeDistributed(Dense(input_shape[1]))(decoded)
    
    # Build the model
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

chosen_train_dataset = 'train_FD003'  # Example dataset to train on
X_train = train_datasets[chosen_train_dataset]['X']
y_train = X_train
input_shape = (X_train.shape[1], X_train.shape[2])  # (sequence_length, number of features)

# Train the model
model = build_cnn_lstm_autoencoder(input_shape)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, shuffle=False, callbacks=[early_stopping])

# Define last N cycles as anomalous - CORRECTED FUNCTION
def create_true_labels_for_sequences(df, sequence_length, anomaly_window):
    # if 'max_cycle' not in df.columns:
    #     raise ValueError("DataFrame must have 'max_cycle' column calculated.")
    if 'time_in_cycles' not in df.columns:
        raise ValueError("DataFrame must have 'time_in_cycles' column.")
    
    df = df.copy()
    df['max_cycle'] = df.groupby('unit_number')['time_in_cycles'].transform('max')
    df['cycles_to_failure'] = df['max_cycle'] - df['time_in_cycles']
    df['true_anomaly'] = (df['cycles_to_failure'] < anomaly_window).astype(int)
    
    true_labels = []
    cycles_to_failure = []
    for unit in sorted(df['unit_number'].unique()):
        unit_data = df[df['unit_number'] == unit]
        unit_labels = unit_data['true_anomaly'].values
        unit_c2f = unit_data['cycles_to_failure'].values
        unit_cycles = unit_data['time_in_cycles'].values
        
        if len(unit_data) >= sequence_length:
            # For each sequence, label is at the END of sequence
            for i in range(len(unit_data) - sequence_length + 1):
                end_idx = i + sequence_length - 1
                true_labels.append(unit_labels[end_idx])
                cycles_to_failure.append(unit_c2f[end_idx])
                
    return np.array(true_labels), np.array(cycles_to_failure)

#reconstruct the sequences
chosen_RUL_dataset = 'RUL_FD003'  # Example RUL dataset to use
anomaly_window = 30  # Define the window for anomalies. You can set 20,30, or 40
chosen_test_dataset = 'test_FD003'  # Example dataset to test on
# Calculate the threshold from the TRAINING data (healthy portion only)
train_df = train_datasets[chosen_train_dataset]['df'].copy()

# The healthy sequences are those that END in the healthy phase
sequence_end_indices = np.arange(sequence_length - 1, len(train_df))
healthy_mask_threshold = train_df['cycle_norm'] <= 0.7
healthy_sequence_mask = healthy_mask_threshold.iloc[sequence_end_indices].values

# Filter the training MSE to only include healthy sequences
train_reconstructions = model.predict(X_train)
train_mse = np.mean((X_train - train_reconstructions) ** 2, axis=(1, 2))
print("Min MSE:", train_mse.min())
print("Max MSE:", train_mse.max())
print("Mean MSE:", train_mse.mean())
healthy_train_mse = train_mse[healthy_sequence_mask]




# Create proper true labels - CORRECT WAY
true_labels, cycles_to_failure = create_true_labels_for_sequences(train_df, sequence_length, anomaly_window)

# Ensure alignment
if len(train_mse) != len(true_labels):
    min_len = min(len(train_mse), len(true_labels))
    train_mse = train_mse[:min_len]
    true_labels = true_labels[:min_len]
    cycles_to_failure = cycles_to_failure[:min_len]
    print(f"Aligned lengths to: {min_len}")

# Calculate the threshold from the healthy data
threshold = np.mean(healthy_train_mse) + 1.2 * np.std(healthy_train_mse)

# Classify test sequences as Normal (0) or Anomalous (1)
train_predictions = (train_mse > threshold).astype(int)

print(f"Anomaly detection threshold: {threshold:.4f}")
print(f"Number of anomalies detected in train set: {np.sum(train_predictions)}")

X_test = test_datasets[chosen_test_dataset]['X']
reconstruction = model.predict(X_test)
test_mse = np.mean(np.square(X_test - reconstruction), axis=(1, 2))

def generate_test_set_labels_and_c2f(test_datasets, rul_datasets, sequence_length, anomaly_window):
    test_df_full = test_datasets[chosen_test_dataset]['df']
    rul_df = rul_datasets[chosen_RUL_dataset]
    unique_units = sorted(test_df_full['unit_number'].unique())

    
    all_true_labels = []
    all_cycles_to_failure = []
    
    for unit in unique_units:
        unit_mask = test_df_full['unit_number'] == unit
        unit_data = test_df_full[unit_mask].copy()
        num_test_cycles = len(unit_data)
        unit_data['max_cycle'] = unit_data['time_in_cycles'].max()
        unit_data['cycles_to_failure'] = unit_data['max_cycle'] - unit_data['time_in_cycles']
        
        if num_test_cycles < sequence_length:
            print(f"Unit {unit} has insufficient data for sequences")
            continue
        
        # Get the RUL for this unit from the RUL dataframe
        rul_value = rul_df.iloc[unit - 1]['RUL']
        max_cycle = num_test_cycles + rul_value
        unit_data['max_cycle'] = max_cycle
        
        try:
            unit_true_labels, unit_c2f = create_true_labels_for_sequences(unit_data, sequence_length, anomaly_window)
            all_true_labels.extend(unit_true_labels)
            all_cycles_to_failure.extend(unit_c2f)
        except ValueError as e:
            print(f"Error processing unit {unit}: {e}")
    
    if all_true_labels and all_cycles_to_failure:
        test_true_labels = np.array(all_true_labels)
        test_cycles_to_failure = np.array(all_cycles_to_failure)
        return test_true_labels, test_cycles_to_failure
    else:
        return np.array([]), np.array([])
    
test_true_labels, test_cycles_to_failure_end = generate_test_set_labels_and_c2f(
    test_datasets, rul_datasets, sequence_length, anomaly_window
)


#calculate early detection rate
def calculate_early_detection_rate(true_labels, predictions, cycles_to_failure, horizon):
    anomaly_indices = np.where(true_labels == 1)[0]
    early_detections = 0
    for idx in anomaly_indices:
        if predictions[idx] == 1 and cycles_to_failure[idx] <= horizon:
            early_detections += 1
    total_actual_anomalies = len(anomaly_indices)
    return early_detections / total_actual_anomalies if total_actual_anomalies > 0 else 0

def calculate_false_positive_rate(true_labels, predictions):
    normal_indices = np.where(true_labels == 0)[0]
    false_alarms = 0
    for idx in normal_indices:
        if predictions[idx] == 1:
            false_alarms += 1
    total_normal = len(normal_indices)
    return false_alarms / total_normal if total_normal > 0 else 0


if len(test_mse) > 0 and len(test_true_labels) > 0:
    # Ensure alignment (should be okay if generated correctly)
    min_test_len = min(len(test_mse), len(test_true_labels), len(test_cycles_to_failure_end))
    test_mse_aligned = test_mse[:min_test_len]
    test_true_labels_aligned = test_true_labels[:min_test_len]
    test_cycles_to_failure_end_aligned = test_cycles_to_failure_end[:min_test_len]

    test_predictions = (test_mse_aligned > threshold).astype(int)

    # 4. Calculate Metrics on Test Set
    horizon = 20 # Define your horizon
    # Ensure calculate_early_detection_rate and calculate_false_positive_rate are corrected
    # e.g., calculate_early_detection_rate should calculate anomaly_indices internally
    # and use cycles_to_failure[idx] >= horizon for early detection logic

    edr_test = calculate_early_detection_rate(test_true_labels_aligned, test_predictions, test_cycles_to_failure_end_aligned, horizon)
    fpr_test = calculate_false_positive_rate(test_true_labels_aligned, test_predictions)

    print(f"\n--- Test Set Performance Metrics (Threshold: {threshold:.4f}) ---")
    print(f"Early Detection Rate ({horizon} cycles early): {edr_test:.4f}")
    print(f"False Positive Rate: {fpr_test:.4f}")
    print(f"Total Test Sequences: {len(test_true_labels_aligned)}")
    print(f"Actual Anomalies in Test Set: {np.sum(test_true_labels_aligned)}")
    print(f"Predicted Anomalies in Test Set: {np.sum(test_predictions)}")

else:
    print("Could not generate test set labels or predictions due to insufficient data or errors.")



horizon = 15  # Define how many cycles before failure we consider for early detection
edr_train = calculate_early_detection_rate(true_labels, train_predictions, cycles_to_failure, horizon)
fpr_train = calculate_false_positive_rate(true_labels, train_predictions)
print(f"Early Detection Rate (EDR) on training set: {edr_train:.4f}")
print(f"False Positive Rate (FPR) on training set: {fpr_train:.4f}")

# Comprehensive evaluation
print(f"\nClass Distribution:")
print(f"Normal samples: {np.sum(true_labels == 0)}")
print(f"Anomalous samples: {np.sum(true_labels == 1)}")
print(f"Anomaly ratio: {np.mean(true_labels):.4f}")



percentiles = [90, 92, 95, 97, 99]
results = []

for p in percentiles:
    threshold = np.percentile(healthy_train_mse, p)
    train_predictions = (train_mse > threshold).astype(int)
   
    anomalies_detected = np.sum(train_predictions)
    results.append((p, threshold, anomalies_detected))
    print(f"Percentile: {p}, Threshold: {threshold:.4f}, ")
  
#find a specific engine
engine_id = 75
engine_mask = test_datasets[chosen_test_dataset]['df']['unit_number'] == engine_id
engine_df = test_datasets[chosen_test_dataset]['df'][engine_mask]
engine_cycles = engine_df['time_in_cycles'].values

# Calculate sequence start index more reliably
sequence_start_idx = 0
for i in range(1, engine_id):
    prev_engine_mask = test_datasets[chosen_test_dataset]['df']['unit_number'] == i
    prev_engine_cycles = test_datasets[chosen_test_dataset]['df'][prev_engine_mask]['time_in_cycles'].values
    if len(prev_engine_cycles) >= sequence_length:
        sequence_start_idx += len(prev_engine_cycles) - sequence_length + 1

num_sequences = len(engine_cycles) - sequence_length + 1
if num_sequences > 0:
    engine_mse = test_mse[sequence_start_idx:sequence_start_idx + num_sequences]
    x_axis_cycles = engine_cycles[sequence_length - 1:]
else:
    print(f"Engine {engine_id} has insufficient data for sequences")
    engine_mse = np.array([])
    x_axis_cycles = np.array([])

if len(engine_mse) > 0:
    plt.figure(figsize=(12, 6))
    plt.plot(x_axis_cycles, engine_mse, label='Reconstruction Error (MSE)')
    plt.axhline(y=threshold, color='r', linestyle='--', label='Anomaly Threshold')
    plt.xlabel('Time in Cycles')
    plt.ylabel('Reconstruction Error (MSE)')
    plt.title(f'Anomaly Detection for Engine {engine_id} (Test Dataset {chosen_test_dataset})')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Anomaly prediction
    if len(engine_mse) >= 20:  # Need at least 20 points for prediction
        N = min(20, len(engine_mse))
        X = x_axis_cycles[-N:].reshape(-1, 1)
        y = engine_mse[-N:]

        # Polynomial feature expansion
        poly = PolynomialFeatures(degree=2)
        X_poly = poly.fit_transform(X)

        # Train a polynomial regression model
        poly_model = Ridge(alpha=1.0)
        poly_model.fit(X_poly, y)

        # Predict on historical data to visualize fit
        y_pred = poly_model.predict(X_poly)

        # Predict the next M cycles
        M = 50
        X_future = np.arange(x_axis_cycles[-1] + 1, x_axis_cycles[-1] + M + 1).reshape(-1, 1)
        X_future_poly = poly.transform(X_future)
        y_future = poly_model.predict(X_future_poly)

        plt.figure(figsize=(12, 6))
        plt.plot(x_axis_cycles, engine_mse, label='Reconstruction Error (MSE)')
        plt.plot(x_axis_cycles[-N:], y_pred, label='Model Fit (Last 20 Cycles)', color='green', linestyle='-', linewidth=2)
        plt.plot(X_future.flatten(), y_future, label='Predicted Future MSE', color='orange')
        plt.axhline(y=threshold, color='r', linestyle='--', label='Anomaly Threshold')
        
        # Find when the future MSE exceeds the threshold
        anomaly_indices = np.where(y_future > threshold)[0]
        if len(anomaly_indices) > 0:
            anomaly_cycle = X_future[anomaly_indices[0]][0]
            print(f"Anomaly predicted at cycle: {anomaly_cycle}")
            plt.axvline(x=anomaly_cycle, color='g', linestyle='--', label='Predicted Anomaly Cycle')
        else:
            print("No anomaly predicted in the next 50 cycles")
        plt.xlabel('Time in Cycles')
        plt.ylabel('Reconstruction Error (MSE)')
        plt.title(f'Anomaly Prediction for Engine {engine_id} (Test Dataset {chosen_test_dataset})')
        plt.legend()
        plt.grid(True)
        plt.show()

# Assuming 'history' is the object returned by model.fit()
# Plot training & validation loss values
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss (MSE)')
plt.plot(history.history['val_loss'], label='Validation Loss (MSE)')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.grid(True)
plt.show()

# --- STEP 1: Calculate the RUL Targets (The "Answers") ---
train_df = train_datasets[chosen_train_dataset]['df'].copy()

# Calculate RUL with 145 Piecewise Cap
train_max_cycle = train_df.groupby('unit_number')['time_in_cycles'].max().reset_index()
train_max_cycle.columns = ['unit_number', 'max_cycle']
train_df = train_df.merge(train_max_cycle, on='unit_number', how='left')
train_df['RUL_raw'] = train_df['max_cycle'] - train_df['time_in_cycles']

MAX_RUL = 155 
train_df['RUL'] = train_df['RUL_raw'].clip(upper=MAX_RUL)
train_targets = train_df['RUL'].values[sequence_length - 1:]

# --- STEP 2: Extract Latent Features & Inject Time ---
# Extracting 12D vectors from the Autoencoder bottleneck (index 11)
encoder_model = Model(inputs=model.input, outputs=model.get_layer(index=11).output)
train_latent = encoder_model.predict(X_train)
test_latent = encoder_model.predict(X_test)

# **THE KEY IMPROVEMENT**: Normalizing and adding 'Time' as a feature
# This gives the model a 'gradient' to follow so the red line isn't flat.
train_time = (train_df['time_in_cycles'].values[sequence_length - 1:] / 300).reshape(-1, 1)
test_df = test_datasets[chosen_test_dataset]['df']
test_time = (test_df['time_in_cycles'].values[sequence_length - 1:] / 300).reshape(-1, 1)

train_features = np.hstack([train_latent, train_time])
test_features = np.hstack([test_latent, test_time])

import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.keras import layers

def critical_zone_loss(y_true, y_pred):
    """
    Custom loss function for RUL regression that penalizes overestimation more heavily,
    especially as the Remaining Useful Life (RUL) approaches zero (critical zone).
    The penalty increases exponentially for overestimation, and the weighting gives
    higher importance to samples with lower RUL to encourage accurate predictions near failure.
    """
    # squarred_error = K.square(y_true - y_pred)
    delta = y_true - y_pred
    mse = K.square(delta)
    diff = y_pred - y_true
    penalty = tf.where(
        diff > 0,
        K.minimum(K.exp(diff/30), 5.0),  # More penalty for overestimation
        K.ones_like(diff)
    )  # More penalty for overestimation
    weight = 20.0 / (y_true + 10)  # More weight on smaller RUL values
    return K.mean(weight * penalty * mse)
# --- STEP 3: Build and Train the Sensitive Regressor ---

    
#     # 3. Lower learning rate (0.0005) gives the model more time to 
#     # find the subtle 'wear' patterns instead of jumping to the average.
#     optimizer = Adam(learning_rate=0.0002)
#     model.compile(optimizer=optimizer, loss=critical_zone_loss, metrics=['mae'])
#     return model

def build_sensitive_regressor(input_dim):
    
    inputs = Input(shape=(input_dim,))
    
    # Main path
    x = Dense(256, activation='swish')(inputs)
    x = BatchNormalization()(x)
    
    # Add a skip connection (Residual) to stabilize gradients
    res1 = Dense(128, activation='linear')(x) 
    x = Dense(128, activation='swish')(x)
    x = Add()([x, res1]) # Residual link
    x = BatchNormalization()(x)
    
    res2 = Dense(64, activation='linear')(x)
    x = Dense(64, activation='swish')(x)
    output = Dense(1, activation='sigmoid')(x) 
    prediction = Lambda(lambda x: x*155.0)(output)  # Linear activation for regression
    model = Model(inputs=inputs, outputs=prediction)
    optimizer = Adam(learning_rate=0.0002)
    model.compile(optimizer=optimizer, loss=critical_zone_loss, metrics=['mae'])
    return model



# Add this to your .fit() call to help it converge better
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5)

rul_regressor = build_sensitive_regressor(train_features.shape[1])
history_rul = rul_regressor.fit(
    train_features, train_targets, 
    epochs=100, batch_size=32, validation_split=0.2, callbacks=[reduce_lr]
)

# --- STEP 4: Evaluation & Smoothing ---
test_predictions_raw = rul_regressor.predict(test_features)

# Smoothing function to clean up sensor jitter
def smooth_preds(preds, window=7):
    return pd.Series(preds.flatten()).rolling(window=window, min_periods=1).mean().values

test_predictions_smooth = smooth_preds(test_predictions_raw)

# --- STEP 5: Metrics & Score ---
unique_units = test_df['unit_number'].unique()
last_predictions = []
current_idx = 0

for unit in unique_units:
    unit_data = test_df[test_df['unit_number'] == unit]
    num_sequences = len(unit_data) - sequence_length + 1
    if num_sequences > 0:
        # We use the smoothed prediction for the final metric
        last_predictions.append(test_predictions_smooth[current_idx + num_sequences - 1])
        current_idx += num_sequences

actual_rul = rul_datasets['RUL_FD004']['RUL'].values[:len(last_predictions)]
rmse = np.sqrt(mean_squared_error(actual_rul, last_predictions))

def cmapss_score(y_true, y_pred):
    diff = y_pred - y_true
    return np.sum([np.exp(-d/13)-1 if d<0 else np.exp(d/10)-1 for d in diff])

print(f"Final RMSE: {rmse:.4f}")
print(f"C-MAPSS Score: {cmapss_score(actual_rul, np.array(last_predictions)):.2f}")

# --- STEP 6: Visualization (Unit 50) ---
sample_unit = 50
unit_mask = test_df['unit_number'] == sample_unit
unit_seq_count = len(test_df[unit_mask]) - sequence_length + 1

# Calculate plot offset
offset = 0
for unit in unique_units:
    if unit == sample_unit: break
    u_len = len(test_df[test_df['unit_number'] == unit])
    if u_len >= sequence_length: offset += (u_len - sequence_length + 1)

sample_preds = test_predictions_smooth[offset : offset + unit_seq_count]
true_rul_val = rul_datasets['RUL_FD004']['RUL'].values[sample_unit-1]
true_line = np.arange(unit_seq_count)[::-1] + true_rul_val

plt.figure(figsize=(12, 6))
plt.plot(true_line, label='Ground Truth', color='blue', linestyle='--')
plt.plot(sample_preds, label='Time-Aware Prediction', color='darkred', linewidth=2)
plt.axvspan(120, unit_seq_count, color='green', alpha=0.1, label='Critical Failure Zone')
plt.title(f'Optimized RUL: Engine {sample_unit} (FD004)')
plt.ylabel('RUL (Cycles)')
plt.xlabel('Time (Cycles)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()