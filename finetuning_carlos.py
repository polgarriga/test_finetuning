from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModel, DataCollatorForSeq2Seq
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer

import pandas as pd


def dataframe_to_hf(df,src,tgt):
    translations = []
    for s,t in zip(df[src],df[tgt]):
        translations.append({src:s,tgt:t})
    return {'id':range(len(translations)), 'translation':translations}

checkpoint = "facebook/m2m100_418M"

model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

ds = load_dataset("csv", data_files="corpus_sm.csv")

#load dataset in HuggingFace format
dataset = {}
train_hf_split = dataframe_to_hf(ds['train'][:200],'de','ca')
valid_hf_split = dataframe_to_hf(ds['train'][200:250],'de','ca')
#print(test_hf_split)
train_hf_split = Dataset.from_dict(train_hf_split)
valid_hf_split = Dataset.from_dict(valid_hf_split)
dataset = DatasetDict({"train": train_hf_split, "valid":valid_hf_split})

tokenizer = AutoTokenizer.from_pretrained(checkpoint, return_tensors="tf", src_lang="de", tgt_lang="ca")

source_lang = "de"
target_lang = "ca"

def preprocess_function(examples):
    inputs = [example[source_lang] for example in examples["translation"]]
    targets = [example[target_lang] for example in examples["translation"]]
    model_inputs = tokenizer(inputs, max_length=128, truncation=True)

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=128, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = dataset.map(preprocess_function, batched=True)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    save_total_limit=1,
    num_train_epochs=1,
    fp16=False,
    save_strategy="epoch",
    logging_strategy="epoch",
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["valid"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()
