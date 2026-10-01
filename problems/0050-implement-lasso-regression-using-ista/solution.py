import numpy as np

def soft_threshold(w: np.ndarray, threshold: float) -> np.ndarray:
    """Apply soft-thresholding operator element-wise.
    
    S(w, λ) = sign(w) * max(|w| - λ, 0)
    """
    return np.sign(w) * np.maximum(np.abs(w) - threshold, 0)


def l1_regularization_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float = 0.1,
    learning_rate: float = 0.01,
    max_iter: int = 1000,
    tol: float = 1e-4
) -> tuple:
    """
    Implement Lasso Regression using ISTA.
    """

    n_samples, n_features = X.shape

    weights = np.zeros(n_features)
    bias = 0.0

    for _ in range(max_iter):

        # Predictions
        y_pred = X @ weights + bias

        # Error
        error = y_pred - y

        # Gradient of MSE with respect to weights
        grad_w = (X.T @ error) / n_samples

        # Gradient of MSE with respect to bias
        grad_b = np.mean(error)

        # Gradient descent step
        weights_temp = weights - learning_rate * grad_w
        bias_new = bias - learning_rate * grad_b

        # Proximal step / soft thresholding
        weights_new = soft_threshold(
            weights_temp,
            learning_rate * alpha
        )

        # Check convergence
        if np.linalg.norm(weights_new - weights) < tol:
            weights = weights_new
            bias = bias_new
            break

        weights = weights_new
        bias = bias_new

    return weights, bias