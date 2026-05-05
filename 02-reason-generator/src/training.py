from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments


def create_training_arguments(epochs, train_batch_size, learning_rate, output_dir, max_length):
    return Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=train_batch_size,
        learning_rate=learning_rate,
        predict_with_generate=True,
        generation_max_length=max_length,
        eval_strategy="epoch",
        save_strategy="no",
    )


def create_trainer(model, train_args, train_dataset, dev_dataset, tokenizer):
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    return Seq2SeqTrainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=data_collator,
    )
