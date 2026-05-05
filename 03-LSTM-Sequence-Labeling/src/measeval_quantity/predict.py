from typing import List

import numpy as np
from tensorflow.keras.models import Sequential


def make_predictions(model: Sequential, x_test: np.ndarray, batch_size: int) -> np.ndarray:
    y_pred = model.predict(x_test, batch_size=batch_size)
    y_pred = np.argmax(y_pred, axis=-1)
    return y_pred


def predictions_to_labels(predictions: np.ndarray, x_test: np.ndarray, labels: List[str]) -> List[str]:
    pred_labels: List[str] = []
    for pred_seq, x_seq in zip(predictions, x_test):
        pred_seq_labels = [labels[p] for p, x in zip(pred_seq, x_seq) if x != 0]
        pred_labels.extend(pred_seq_labels)
    return pred_labels
