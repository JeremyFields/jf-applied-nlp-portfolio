from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingConfig:
    model_name: str = "facebook/bart-base"
    max_length: int = 25
    epochs: int = 3
    train_batch_size: int = 8
    learning_rate: float = 1e-5
    output_dir: str = "models/commonsense-reason-generator"
    seed: int = 42
