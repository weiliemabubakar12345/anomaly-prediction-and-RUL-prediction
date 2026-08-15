import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from xjtu_autoencoder_common import load_xjtu_windows, save_training_history, plot_mse_distribution, save_reconstruction_examples, save_summary

out_dir = r'D:\Springer_Nature_LaTeX_Template\Paper 1 _ Preventive Maintenance\results\CNN_LSTM_Autoencoder_v2'
os.makedirs(out_dir, exist_ok=True)

X_train, X_test, _, _, _ = load_xjtu_windows(window_len=128, step=128, max_windows=8000)
input_shape = X_train.shape[1:]
num_features = input_shape[-1]

inputs = tf.keras.Input(shape=input_shape)
conv1 = layers.Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(inputs)
pool1 = layers.MaxPooling1D(pool_size=2, padding='same')(conv1)
conv2 = layers.Conv1D(filters=16, kernel_size=3, activation='relu', padding='same')(pool1)
pool2 = layers.MaxPooling1D(pool_size=2, padding='same')(conv2)
encoded = layers.LSTM(64, activation='tanh', return_sequences=True)(pool2)
encoded = layers.Dropout(0.2)(encoded)
encoded = layers.LSTM(32, activation='tanh', return_sequences=True)(encoded)
encoded = layers.Dropout(0.2)(encoded)
encoded = layers.LSTM(16, activation='tanh')(encoded)
decoded = layers.RepeatVector(input_shape[0])(encoded)
decoded = layers.LSTM(16, activation='tanh', return_sequences=True)(decoded)
decoded = layers.LSTM(32, activation='tanh', return_sequences=True)(decoded)
decoded = layers.LSTM(64, activation='tanh', return_sequences=True)(decoded)
outputs = layers.TimeDistributed(layers.Dense(num_features))(decoded)
model = Model(inputs, outputs)
model.compile(optimizer=Adam(learning_rate=1e-3), loss='mse')
model.summary()

callbacks = [EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]
history = model.fit(X_train, X_train, epochs=8, batch_size=32, validation_split=0.1, shuffle=True, callbacks=callbacks)

save_training_history(history, os.path.join(out_dir, 'training_loss.png'))

train_recon = model.predict(X_train, verbose=0)
train_mse = np.mean(np.square(X_train - train_recon), axis=(1, 2))

test_recon = model.predict(X_test, verbose=0)
test_mse = np.mean(np.square(X_test - test_recon), axis=(1, 2))

threshold = np.mean(train_mse) + 3 * np.std(train_mse)
plot_mse_distribution(train_mse, test_mse, threshold, os.path.join(out_dir, 'mse_distribution.png'))

anomaly_idxs = np.where(test_mse > threshold)[0]
print(f'CNN-LSTM Autoencoder anomalous test windows: {len(anomaly_idxs)}')
if len(anomaly_idxs):
    save_reconstruction_examples(X_test[:len(anomaly_idxs)], test_recon[:len(anomaly_idxs)], test_mse[:len(anomaly_idxs)], os.path.join(out_dir, 'anom_examples'), prefix='cnn_lstm_test', max_examples=min(5, len(anomaly_idxs)))

save_summary(
    os.path.join(out_dir, 'results_summary.txt'),
    model='CNN-LSTM Autoencoder',
    input_shape=input_shape,
    train_windows=X_train.shape[0],
    test_windows=X_test.shape[0],
    threshold=float(threshold),
    train_mse_mean=float(np.mean(train_mse)),
    train_mse_std=float(np.std(train_mse)),
    test_mse_mean=float(np.mean(test_mse)),
    anomalous_test_windows=int(len(anomaly_idxs)),
    num_anomalous_examples=int(min(5, len(anomaly_idxs)))
)

print('CNN-LSTM Autoencoder run done.')
