import numpy as np
import torch

def compute_entropy_uncertainty_numpy(pred):
    """
    Computes entropy uncertainty of the model's output probability map (NumPy version).

    Args:
        pred (np.ndarray): Probability map of shape [B, C, H, W], where class probabilities sum to 1 at each pixel.

    Returns:
        entropy (np.ndarray): Entropy uncertainty map of shape [B, H, W].
    """
    epsilon = 1e-8  # To prevent log(0)
    pred = np.clip(pred, epsilon, 1.0)  # Ensure numerical stability, avoid log(0)
    entropy = -np.sum(pred * np.log(pred), axis=1)  # Compute entropy along the class dimension, resulting shape [B, H, W]
    return entropy

def compute_entropy_uncertainty_torch(pred):
    """
    Computes entropy uncertainty of the model's output probability map (PyTorch version).

    Args:
        pred (torch.Tensor): Probability map of shape [B, C, H, W], where class probabilities sum to 1 at each pixel.

    Returns:
        entropy (torch.Tensor): Entropy uncertainty map of shape [B, H, W].
    """
    epsilon = 1e-8  # To prevent log(0)
    pred = torch.clamp(pred, epsilon, 1.0)  # Ensure numerical stability, avoid log(0)
    entropy = -torch.sum(pred * torch.log(pred + epsilon), dim=1)  # Add epsilon to prevent log(0)
    return entropy


def UDAT(uncertainty_maps: torch.Tensor, k: float = 0.2) -> float:
    """
    Calculates a unified threshold for the batch by computing the threshold for each image
    based on mean + standard deviation, and then averaging these thresholds.

    Args:
        uncertainty_maps (torch.Tensor): Uncertainty map of shape [B, HW].
        k (float): Scaling coefficient, default is 0.2.

    Returns:
        float: The unified threshold for the batch.
    """
    # Calculate the mean and standard deviation of uncertainty for each image, shape is [B]
    means = torch.mean(uncertainty_maps, dim=1)  # [B]
    stds = torch.std(uncertainty_maps, dim=1)    # [B]

    # Threshold for each image = mean + k * standard deviation, shape [B]
    thresholds = means + k * stds

    # Average the thresholds to get the unified batch threshold
    batch_threshold = thresholds.mean()

    return batch_threshold.item()
