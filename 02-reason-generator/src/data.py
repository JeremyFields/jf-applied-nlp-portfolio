import pandas as pd
from datasets import Dataset


def load_data(data_csv, answers_csv, is_test=False):
    data = pd.read_csv(data_csv).dropna()
    answers = pd.read_csv(answers_csv, header=None).rename(columns={0: "id", 1: "reason1", 2: "reason2", 3: "reason3"})
    if is_test:
        return pd.merge(data, answers, on="id")
    answers = answers.melt(id_vars="id", value_name="reason")[["id", "reason"]]
    return pd.merge(data, answers, on="id")


def preprocess_data(examples, tokenizer, max_length, is_test=False):
    if is_test:
        return tokenizer(
            examples["FalseSent"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    model_inputs = tokenizer(
        examples["FalseSent"],
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )
    labels = tokenizer(
        examples["reason"],
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def prepare_datasets(train_data, dev_data, test_data, tokenizer, max_length):
    train_dataset = Dataset.from_pandas(train_data)
    dev_dataset = Dataset.from_pandas(dev_data)
    test_dataset = Dataset.from_pandas(test_data)

    train_dataset = train_dataset.map(
        lambda batch: preprocess_data(batch, tokenizer, max_length),
        batched=True,
        remove_columns=train_dataset.column_names,
    )
    dev_dataset = dev_dataset.map(
        lambda batch: preprocess_data(batch, tokenizer, max_length),
        batched=True,
        remove_columns=dev_dataset.column_names,
    )
    test_dataset = test_dataset.map(
        lambda batch: preprocess_data(batch, tokenizer, max_length, True),
        batched=True,
        remove_columns=test_dataset.column_names,
    )
    return train_dataset, dev_dataset, test_dataset
