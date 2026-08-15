import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from xjtu_autoencoder_common import load_xjtu_windows, save_training_history, plot_mse_distribution, save_reconstruction_examples, save_summary

out_dir = r'D:\Springer_Nature_LaTeX_Template\Paper 1 _ Preventive Maintenance\results\VAE_v2'
os.makedirs(out_dir, exist_ok=True)

X_train, X_test, _, _, _ = load_xjtu_windows(window_len=128, step=128, max_windows=8000)
X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)
input_shape = X_train.shape[1:]
num_features = input_shape[-1]
latent_dim = 16

class VAE(Model):
    def __init__(self, latent_dim=latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential([
            layers.Input(shape=input_shape),
            layers.Conv1D(32, 3, padding='same', activation='relu'),
            layers.MaxPooling1D(2, padding='same'),
            layers.Conv1D(16, 3, padding='same', activation='relu'),
            layers.MaxPooling1D(2, padding='same'),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
        ])
        self.z_mean = layers.Dense(latent_dim)
        self.z_log_var = layers.Dense(latent_dim)
        self.decoder = tf.keras.Sequential([
            layers.Input(shape=(latent_dim,)),
            layers.Dense(64, activation='relu'),
            layers.Dense(128 * num_features, activation='linear'),
            layers.Reshape((128, num_features)),
        ])

    def encode(self, x):
        h = self.encoder(x)
        z_mean = self.z_mean(h)
        z_log_var = self.z_log_var(h)
        return z_mean, z_log_var

    def reparameterize(self, z_mean, z_log_var):
        eps = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * eps

    def decode(self, z):
        return self.decoder(z)

    def call(self, x):
        z_mean, z_log_var = self.encode(x)
        z = self.reparameterize(z_mean, z_log_var)
        x_decoded = self.decode(z)
        return x_decoded, z_mean, z_log_var

vae = VAE()
optimizer = Adam(learning_rate=1e-3)


def vae_loss(x, x_decoded, z_mean, z_log_var):
    x = tf.cast(x, tf.float32)
    x_decoded = tf.cast(x_decoded, tf.float32)
    reconstruction_loss = tf.reduce_mean(tf.reduce_sum(tf.square(x - x_decoded), axis=(1, 2)))
    kl_loss = -0.5 * tf.reduce_mean(tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1))
    return reconstruction_loss + 1e-3 * kl_loss

train_loss_history = []
val_loss_history = []
for epoch in range(8):
    for batch in tf.data.Dataset.from_tensor_slices(X_train).batch(64):
        with tf.GradientTape() as tape:
            x_decoded, z_mean, z_log_var = vae(batch)
            loss = vae_loss(batch, x_decoded, z_mean, z_log_var)
        grads = tape.gradient(loss, vae.trainable_variables)
        optimizer.apply_gradients(zip(grads, vae.trainable_variables))
    val_batch = X_train[: int(len(X_train) * 0.1)]
    val_decoded, z_mean, z_log_var = vae(val_batch)
    val_loss = vae_loss(val_batch, val_decoded, z_mean, z_log_var)
    sample_out = vae(X_train[:128])
    train_loss_history.append(float(vae_loss(X_train[:128], sample_out[0], sample_out[1], sample_out[2]).numpy()))
    val_loss_history.append(float(val_loss.numpy()))

history = type('H', (), {'history': {'loss': train_loss_history, 'val_loss': val_loss_history}})()
save_training_history(history, os.path.join(out_dir, 'training_loss.png'))

train_recon, _, _ = vae(X_train)
train_mse = np.mean(np.square(X_train - train_recon), axis=(1, 2))

test_recon, _, _ = vae(X_test)
test_mse = np.mean(np.square(X_test - test_recon), axis=(1, 2))

threshold = np.mean(train_mse) + 3 * np.std(train_mse)
plot_mse_distribution(train_mse, test_mse, threshold, os.path.join(out_dir, 'mse_distribution.png'))

anomaly_idxs = np.where(test_mse > threshold)[0]
print(f'VAE anomalous test windows: {len(anomaly_idxs)}')
if len(anomaly_idxs):
    save_reconstruction_examples(X_test[:len(anomaly_idxs)], test_recon[:len(anomaly_idxs)], test_mse[:len(anomaly_idxs)], os.path.join(out_dir, 'anom_examples'), prefix='vae_test', max_examples=min(5, len(anomaly_idxs)))

save_summary(
    os.path.join(out_dir, 'results_summary.txt'),
    model='Variational Autoencoder',
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

print('VAE run done.')
