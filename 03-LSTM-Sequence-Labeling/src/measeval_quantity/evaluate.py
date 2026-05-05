import pandas as pd
from sklearn.metrics import classification_report


def evaluate_quantity(data: pd.DataFrame) -> str:
    labels = data.apply(lambda x: x["label"].replace("B-", "").replace("I-", ""), axis=1).values
    predictions = data.apply(lambda x: x["prediction"].replace("B-", "").replace("I-", ""), axis=1).values
    return classification_report(labels, predictions, labels=["Quantity"])
