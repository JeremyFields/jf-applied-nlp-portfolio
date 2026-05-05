from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences


def get_vocabulary(train_data: pd.DataFrame) -> Tuple[List[str], Dict[str, int], List[str], Dict[str, int]]:
    unique_lemmas = train_data["lemma"].unique().tolist()
    vocab = ["[PAD]", "[UNK]"] + unique_lemmas

    unique_labels = train_data["label"].unique().tolist()
    labels = ["[PAD]"] + unique_labels

    word2idx = {word: i for i, word in enumerate(vocab)}
    label2idx = {label: i for i, label in enumerate(labels)}
    return vocab, word2idx, labels, label2idx


def integrate_sentences(data: pd.DataFrame) -> pd.DataFrame:
    agg_func = lambda s: [s["lemma"].values.tolist(), s["label"].values.tolist()]
    grouped = (
        data.groupby(["docId", "sentId"], sort=False)
        .apply(agg_func, include_groups=False)
        .reset_index()
        .rename(columns={0: "lemmas_labels"})
    )
    grouped["lemmas"] = grouped.apply(lambda x: x["lemmas_labels"][0], axis=1)
    grouped["labels"] = grouped.apply(lambda x: x["lemmas_labels"][1], axis=1)
    grouped = grouped.drop(columns="lemmas_labels")
    return grouped


def format_examples(
    data: pd.DataFrame,
    word2idx: Dict[str, int],
    label2idx: Dict[str, int],
    maxlen: int,
) -> Tuple[np.ndarray, np.ndarray]:
    x = [
        [word2idx.get(word, word2idx["[UNK]"]) for word in sentence]
        for sentence in data["lemmas"]
    ]
    y = [[label2idx[label] for label in sentence] for sentence in data["labels"]]

    x = pad_sequences(x, maxlen=maxlen)
    y = pad_sequences(y, maxlen=maxlen)
    return x, y
