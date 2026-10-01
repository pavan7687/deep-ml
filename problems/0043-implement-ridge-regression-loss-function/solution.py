import numpy as np

def ridge_loss(X: np.ndarray, w: np.ndarray, y_true: np.ndarray, alpha: float) -> float:
	# Your code here
	return np.sum((X @ W - y_true)**2/X.shape[0]) + np.sum(W**2)*alpha
