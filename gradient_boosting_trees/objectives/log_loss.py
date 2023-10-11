import numpy as np
from scipy.special import expit as sigmoid

def log_loss_derivative(raw_predictions: np.ndarray, labels: np.ndarray) -> np.ndarray:
    predictions = sigmoid(raw_predictions)
    return predictions * (labels - predictions)
