from typing import Any

import numpy as np
from tensorflow.keras.models import Sequential


def train_model(
    model: Sequential,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_dev: np.ndarray,
    y_dev: np.ndarray,
    batch_size: int,
    epochs: int,
) -> Any:
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_dev, y_dev),
        batch_size=batch_size,
        epochs=epochs,
    )
    return history
