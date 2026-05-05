from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentConfig:
    maxlen: int = 130
    epochs: int = 5
    batch_size: int = 64
    embedding_dim: int = 300
    rnn_units: int = 256
    seed: int = 42
    shrink_dataset: bool = False
