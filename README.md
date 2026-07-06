# anomaly-prediction-and-RUL-prediction
Including code and explanation

# Incipient Anomaly Forecasting & Conservative RUL Prediction

This repository contains the official implementation pipeline for our integrated deep learning framework evaluated on the NASA C-MAPSS dataset.

## Prerequisites
- Python 3.8+
- TensorFlow 2.x
- Sckit-Learn, Pandas, NumPy, Matplotlib, Seaborn

## Usage
1. Download the turbofan engine degradation simulation datasets (FD001 - FD004) from the NASA Prognostics Data Repository. In this directory, same dataset is also provided for convinience.
2. Place the `.txt` files into a subfolder named `Dataset/`.
3. Run the pipeline script:
   ```bash
   python "Anomaly prediction_CNN LSTM Autoencoder+Polynomial (1).py"
