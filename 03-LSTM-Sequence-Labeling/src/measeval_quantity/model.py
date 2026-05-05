import numpy as np
from tensorflow.keras.initializers import Constant
from tensorflow.keras.layers import Bidirectional, Dense, Embedding, Input, LSTM, TimeDistributed
from tensorflow.keras.models import Sequential


def create_model(vocab_size: int, label_size: int, maxlen: int, embedding_dim: int, rnn_units: int) -> Sequential:
    model = Sequential()
    model.add(Input(shape=(maxlen,)))
    model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, mask_zero=True))
    model.add(Bidirectional(LSTM(units=rnn_units, return_sequences=True)))
    model.add(TimeDistributed(Dense(label_size, activation="softmax")))
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"],
    )
    return model


def create_model_with_embeddings(
    vocab_size: int,
    label_size: int,
    maxlen: int,
    embedding_dim: int,
    rnn_units: int,
    embedding_matrix: np.ndarray,
) -> Sequential:
    model = Sequential()
    model.add(Input(shape=(maxlen,)))
    model.add(
        Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            embeddings_initializer=Constant(embedding_matrix),
            trainable=True,
            mask_zero=True,
        )
    )
    model.add(Bidirectional(LSTM(units=rnn_units, return_sequences=True)))
    model.add(TimeDistributed(Dense(label_size, activation="softmax")))
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"],
    )
    return model
