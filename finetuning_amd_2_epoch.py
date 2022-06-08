from datasets import load_from_disk
from transformers import AutoTokenizer, DataCollatorForSeq2Seq
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer

CHECKPOINT = "/gpfs/projects/bsc88/huggingface/models/m2m100_418M"
DATASET_LOC = "data/ca-de"
source_lang = "de"
target_lang = "ca"

print(f"Loading the checkpoint {CHECKPOINT}")

model = AutoModelForSeq2SeqLM.from_pretrained(CHECKPOINT)
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, return_tensors="pt", src_lang=source_lang, tgt_lang=target_lang)

print(f"Loading the dataset {DATASET_LOC}")

tokenized_dataset = load_from_disk(DATASET_LOC)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

training_args = Seq2SeqTrainingArguments(
        output_dir="./results/epoch-2",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        weight_decay=0.01,
        save_total_limit=1,
        num_train_epochs=10,
        fp16=False,
        load_best_model_at_end=True,
        save_strategy="steps",
        save_steps=20000,
        evaluation_strategy="steps",
        eval_steps=10000,
        logging_strategy="steps",
     )

trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["valid"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )


latest_cp = 'results/epoch-2/checkpoint-170000'

print(f"Resuming from the checkpoint {latest_cp}")

print("Starting training...")

trainer.train(resume_from_checkpoint=True)
