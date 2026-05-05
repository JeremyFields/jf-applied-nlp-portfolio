import evaluate


def make_predictions(trainer, test_dataset, tokenizer):
    predictions = trainer.predict(test_dataset)
    return tokenizer.batch_decode(predictions.predictions, skip_special_tokens=True)


def evaluate_prediction(test_data, metric_name):
    metric = evaluate.load(metric_name)
    references = test_data[["reason1", "reason2", "reason3"]].values.tolist()
    predictions = test_data["prediction"].tolist()
    return metric.compute(predictions=predictions, references=references)
