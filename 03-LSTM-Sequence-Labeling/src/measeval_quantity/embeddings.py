from typing import Dict

import numpy as np


def load_embeddings(glove_path: str) -> Dict[str, np.ndarray]:
    embedding_index: Dict[str, np.ndarray] = {}
    with open(glove_path, encoding="utf8") as glove_file:
        for line in glove_file:
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, "f", sep=" ")
            embedding_index[word] = coefs
    return embedding_index


def create_embedding_matrix(
    embedding_index: Dict[str, np.ndarray],
    word2idx: Dict[str, int],
    vocab_size: int,
    embedding_dim: int,
) -> np.ndarray:
    embedding_matrix = np.zeros((vocab_size, embedding_dim), dtype=np.float32)
    for word, i in word2idx.items():
        if i >= vocab_size:
            continue
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix
