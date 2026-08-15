import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dropout, BatchNormalization, TimeDistributed, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

base = r'D:\Springer_Nature_LaTeX_Template\XJTU-SY'
output_dir = r'D:\Springer_Nature_LaTeX_Template\Paper 1 _ Preventive Maintenance\results\BiLSTM_v2'
os.makedirs(output_dir, exist_ok=True)

# Parameters
window_len = 128
step = 128  # non-overlapping
batch_size = 32
epochs = 8
validation_split = 0.1
latent_units_1 = 64
latent_units_2 = 32
random_seed = 42

# Load X files
x_files = sorted([f for f in os.listdir(base) if f.startswith('xtr') and f.endswith('.npy')])
y_files = sorted([f for f in os.listdir(base) if f.startswith('ytr') and f.endswith('.npy')])
print('Found x files:', x_files)
print('Found y files:', y_files)

X_parts = []
for f in x_files:
    arr = np.load(os.path.join(base, f))
    X_parts.append(arr)

# Stack samples
X_all = np.vstack(X_parts)  # shape (N_samples, T, channels)
print('X_all shape:', X_all.shape)

# Subsample to a reasonable number of windows to avoid memory/time blow-up
max_windows = 40000
if X_all.shape[0] * X_all.shape[1] > max_windows:
    rng = np.random.RandomState(42)
    idx = rng.choice(len(X_all), size=min(max_windows, len(X_all)), replace=False)
    X_all = X_all[idx]
    print('Subsampled X_all to:', X_all.shape)

# Optionally load y labels (not used for unsupervised)
Y_parts = []
for f in y_files:
    arr = np.load(os.path.join(base, f))
    Y_parts.append(arr)
if Y_parts:
    Y_all = np.vstack(Y_parts)
    print('Y_all shape:', Y_all.shape)
else:
    Y_all = None

# Create windows from each sample
def windows_from_sample(sample, window_len, step):
    T = sample.shape[0]
    windows = []
    for s in range(0, T - window_len + 1, step):
        windows.append(sample[s:s+window_len])
    return np.array(windows)

all_windows = []
for i in range(X_all.shape[0]):
    w = windows_from_sample(X_all[i], window_len, step)
    if w.size:
        all_windows.append(w)

all_windows = np.vstack(all_windows)  # (num_windows, window_len, channels)
print('All windows shape:', all_windows.shape)

# Keep a manageable subset for training
max_windows = 8000
if len(all_windows) > max_windows:
    rng = np.random.RandomState(random_seed)
    idx = rng.choice(len(all_windows), size=max_windows, replace=False)
    all_windows = all_windows[idx]
    print('Sampled to', all_windows.shape, 'windows for model training.')

# Shuffle windows
rng = np.random.RandomState(random_seed)
indices = np.arange(len(all_windows))
rng.shuffle(indices)
all_windows = all_windows[indices]

# Split into train/test windows (80/20)
n = len(all_windows)
train_n = int(0.8 * n)
X_train = all_windows[:train_n]
X_test = all_windows[train_n:]
print('Train windows:', X_train.shape, 'Test windows:', X_test.shape)

# Normalize per channel using scaler fitted on train
nsamples, ntime, nchan = X_train.shape
scaler = StandardScaler()
X_train_reshaped = X_train.reshape(-1, nchan)
scaler.fit(X_train_reshaped)
X_train_norm = scaler.transform(X_train_reshaped).reshape(nsamples, ntime, nchan)

nsamples_t = X_test.shape[0]
X_test_norm = scaler.transform(X_test.reshape(-1, nchan)).reshape(nsamples_t, ntime, nchan)

# Build BiLSTM autoencoder
input_shape = (window_len, nchan)
model = Sequential()
model.add(Bidirectional(LSTM(latent_units_1, activation='tanh', return_sequences=True), input_shape=input_shape))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Bidirectional(LSTM(latent_units_2, activation='tanh', return_sequences=True)))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(TimeDistributed(Dense(nchan)))
model.compile(optimizer=Adam(learning_rate=1e-3), loss='mse')
model.summary()

# Train
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
history = model.fit(X_train_norm, X_train_norm, epochs=epochs, batch_size=batch_size, validation_split=validation_split, callbacks=[early_stopping])

# Save training history plot
plt.figure()
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.legend()
plt.title('Training Loss')
plt.savefig(os.path.join(output_dir, 'training_loss.png'))
plt.close()

# Compute reconstructions and MSE on train and test
recon_train = model.predict(X_train_norm)
train_mse = np.mean(np.square(X_train_norm - recon_train), axis=(1,2))

recon_test = model.predict(X_test_norm)
test_mse = np.mean(np.square(X_test_norm - recon_test), axis=(1,2))

# Threshold based on training healthy distribution
threshold = np.mean(train_mse) + 3*np.std(train_mse)

# Save histograms
plt.figure()
plt.hist(train_mse, bins=100, alpha=0.6, label='train')
plt.hist(test_mse, bins=100, alpha=0.6, label='test')
plt.axvline(threshold, color='r', linestyle='--', label='threshold')
plt.legend()
plt.title('MSE distribution')
plt.savefig(os.path.join(output_dir, 'mse_histogram.png'))
plt.close()

# Identify top anomalous windows in test
anomaly_idx = np.where(test_mse > threshold)[0]
print('Number of anomalous windows in test:', len(anomaly_idx))

# Save top anomalies as plots (first 10), matching the other model outputs
anom_dir = os.path.join(output_dir, 'anom_examples')
os.makedirs(anom_dir, exist_ok=True)
for k, idx in enumerate(anomaly_idx[:10]):
    win = X_test[idx]
    recon = recon_test[idx]
    t = np.arange(window_len)
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    for ch in range(nchan):
        plt.plot(t, win[:, ch], label=f'orig_ch{ch}')
    plt.title(f'Test original window idx {idx} MSE={test_mse[idx]:.6f}')
    plt.xlabel('Time step')
    plt.ylabel('Value')
    plt.legend(loc='upper right', fontsize='small')
    plt.grid(True)

    plt.subplot(2, 1, 2)
    for ch in range(nchan):
        plt.plot(t, recon[:, ch], label=f'recon_ch{ch}', linestyle='--')
    plt.title(f'Test reconstruction window idx {idx}')
    plt.xlabel('Time step')
    plt.ylabel('Value')
    plt.legend(loc='upper right', fontsize='small')
    plt.grid(True)
    plt.tight_layout()
    path = os.path.join(anom_dir, f'bilstm_test_{k}.png')
    plt.savefig(path)
    plt.close()

# Save summary results
with open(os.path.join(output_dir, 'results_BiLSTM_v2.txt'), 'w') as f:
    f.write(f'X_all shape: {X_all.shape}\n')
    f.write(f'all_windows shape: {all_windows.shape}\n')
    f.write(f'train windows: {X_train.shape}\n')
    f.write(f'test windows: {X_test.shape}\n')
    f.write(f'threshold: {threshold}\n')
    f.write(f'num anomalies in test: {len(anomaly_idx)}\n')

print('Saved outputs to', output_dir)
print('Done')
