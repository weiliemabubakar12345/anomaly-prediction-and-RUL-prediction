import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

DATA_DIR = r'D:\Springer_Nature_LaTeX_Template\XJTU-SY'


def load_xjtu_windows(window_len=128, step=128, train_ratio=0.8, random_state=42, max_windows=8000):
    files = sorted([f for f in os.listdir(DATA_DIR) if f.startswith('xtr') and f.endswith('.npy')])
    if not files:
        raise FileNotFoundError(f'No XJTU xtr*.npy files found in {DATA_DIR}')

    stacked = np.vstack([np.load(os.path.join(DATA_DIR, f)) for f in files])
    print(f'Loaded XJTU samples shape: {stacked.shape}')

    def extract_windows(sample):
        windows = []
        for start in range(0, sample.shape[0] - window_len + 1, step):
            windows.append(sample[start:start + window_len])
        return np.array(windows)

    all_windows = []
    for sample in stacked:
        windows = extract_windows(sample)
        if windows.size:
            all_windows.append(windows)
    if not all_windows:
        raise ValueError('No windows extracted from dataset.')

    all_windows = np.vstack(all_windows).astype(np.float32)
    print(f'Extracted windows shape: {all_windows.shape}')

    if max_windows is not None and len(all_windows) > max_windows:
        rng = np.random.RandomState(random_state)
        idx = rng.choice(len(all_windows), size=max_windows, replace=False)
        all_windows = all_windows[idx]
        print(f'Subsampled to {all_windows.shape[0]} windows for training.')

    rng = np.random.RandomState(random_state + 1)
    indices = np.arange(len(all_windows))
    rng.shuffle(indices)
    shuffled = all_windows[indices]

    n_train = int(len(shuffled) * train_ratio)
    X_train = shuffled[:n_train]
    X_test = shuffled[n_train:]

    scaler = StandardScaler()
    X_train_flat = X_train.reshape(-1, X_train.shape[-1])
    scaler.fit(X_train_flat)
    X_train_norm = scaler.transform(X_train_flat).reshape(X_train.shape)
    X_test_norm = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

    return X_train_norm, X_test_norm, scaler, X_train, X_test


def save_training_history(history, save_path):
    plt.figure(figsize=(8, 5))
    plt.plot(history.history.get('loss', []), label='train_loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_mse_distribution(train_mse, test_mse, threshold, save_path):
    plt.figure(figsize=(8, 5))
    plt.hist(train_mse, bins=50, alpha=0.7, label='train_mse')
    plt.hist(test_mse, bins=50, alpha=0.7, label='test_mse')
    plt.axvline(threshold, color='r', linestyle='--', label='threshold')
    plt.xlabel('Reconstruction MSE')
    plt.ylabel('Count')
    plt.title('MSE Distribution')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def save_reconstruction_examples(original, recon, mse_values, save_dir, prefix='sample', max_examples=5):
    os.makedirs(save_dir, exist_ok=True)
    n = min(max_examples, len(original))
    for j in range(n):
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        time = np.arange(original.shape[1])
        for ch in range(original.shape[2]):
            axes[0].plot(time, original[j, :, ch], label=f'orig_ch{ch}')
            axes[1].plot(time, recon[j, :, ch], label=f'recon_ch{ch}', linestyle='--')
        axes[0].set_title(f'{prefix} original, mse={mse_values[j]:.4f}')
        axes[1].set_title(f'{prefix} reconstruction')
        axes[0].legend(loc='upper right', fontsize='small')
        axes[1].legend(loc='upper right', fontsize='small')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{prefix}_{j}.png'))
        plt.close(fig)


def save_summary(path, **kwargs):
    with open(path, 'w') as f:
        for k, v in kwargs.items():
            f.write(f'{k}: {v}\n')
