import numpy as np
from typing import Dict

# def rating_metrics(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
#     predictions = np.asarray(predictions, dtype=np.float32)
#     targets = np.asarray(targets, dtype=np.float32)

#     mse = np.mean((predictions - targets) ** 2)
#     rmse = float(np.sqrt(mse))
#     mae = float(np.mean(np.abs(predictions - targets)))

#     return {
#         "mse": float(mse),
#         "rmse": rmse,
#         "mae": mae,
#     }


def rating_metrics(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    predictions = np.asarray(predictions, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float32)

    # clamp only during evaluation
    predictions = np.clip(predictions, 1.0, 5.0)

    mse = np.mean((predictions - targets) ** 2)
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(predictions - targets)))

    return {
        "mse": float(mse),
        "rmse": rmse,
        "mae": mae,
    }


print("done")
