import pandas as pd


def load_data(data_path: str, shrink_dataset: bool, seed: int) -> pd.DataFrame:
    data = pd.read_csv(data_path, sep="\t", encoding="utf8").dropna()
    if shrink_dataset:
        sample = (
            data[["docId", "sentId"]]
            .drop_duplicates()
            .sample(frac=0.2, random_state=seed)
        )
        data = pd.merge(data, sample, on=["docId", "sentId"])
    return data
