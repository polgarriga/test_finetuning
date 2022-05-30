from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel, DataCollatorForSeq2Seq

checkpoint = "facebook/m2m100_418M"

model = AutoModel.from_pretrained(checkpoint)

ds = load_dataset("csv", data_files="corpus_sm.csv")
eval = load_dataset("csv", data_files="eval.csv")

tokenizer = AutoTokenizer.from_pretrained(checkpoint, return_tensors="tf", src_lang="de", tgt_lang="ca")

max_input_length = 128
max_target_length = 128

def preprocess_function(examples):
    inputs = [ex for ex in examples["ca"]]
    targets = [ex for ex in examples["de"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    # Set up the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_target_length, truncation=True)

    model_inputs["encoder_input_ids"] = labels["input_ids"]

    return model_inputs

tokenized_datasets = ds["train"].map(
    preprocess_function,
    batched=True,
    remove_columns=ds["train"].column_names,
)

tokenized_eval = eval["train"].map(
    preprocess_function,
    batched=True,
    remove_columns=eval["train"].column_names,
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

print(tokenized_datasets[1])

batch = data_collator([tokenized_datasets["train"][i] for i in range(1, 3)])

print(batch)

batch.keys()

from torch.utils.data import DataLoader

tokenized_datasets.set_format("torch")
tokenized_eval.set_format("torch")

train_dataloader = DataLoader(
    tokenized_datasets,
    shuffle=True,
    collate_fn=data_collator,
    batch_size=8
)

eval_dataloader = DataLoader(
    tokenized_eval, 
    collate_fn=data_collator, 
    batch_size=8
)

from transformers import AdamW

optimizer = AdamW(model.parameters(), lr=1e-5)

from accelerate import Accelerator

accelerator = Accelerator()
model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader
)

from transformers import get_scheduler

num_train_epochs = 3
num_update_steps_per_epoch = len(train_dataloader)
num_training_steps = num_train_epochs * num_update_steps_per_epoch

lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

# from huggingface_hub import Repository, get_full_repo_name

# model_name = "marian-finetuned-kde4-en-to-fr-accelerate"
# repo_name = get_full_repo_name(model_name)
# repo_name

output_dir = "/run/media/bscuser/6C19-3138/Thesis/m2m100-de-to-ca-accelerate"
# repo = Repository(output_dir, clone_from=repo_name)


### TRAINING LOOP ###

def postprocess(predictions, labels):
    predictions = predictions.cpu().numpy()
    labels = labels.cpu().numpy()

    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [[label.strip()] for label in decoded_labels]
    return decoded_preds, decoded_labels


from tqdm.auto import tqdm
import torch

progress_bar = tqdm(range(num_training_steps))

for epoch in range(num_train_epochs):
    # Training
    model.train()
    for batch in train_dataloader:
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

    # Evaluation
    model.eval()
    for batch in tqdm(eval_dataloader):
        with torch.no_grad():
            generated_tokens = accelerator.unwrap_model(model).generate(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_length=128,
            )
        labels = batch["labels"]

        # Necessary to pad predictions and labels for being gathered
        generated_tokens = accelerator.pad_across_processes(
            generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
        )
        labels = accelerator.pad_across_processes(labels, dim=1, pad_index=-100)

        predictions_gathered = accelerator.gather(generated_tokens)
        labels_gathered = accelerator.gather(labels)

        decoded_preds, decoded_labels = postprocess(predictions_gathered, labels_gathered)
        metric.add_batch(predictions=decoded_preds, references=decoded_labels)

    results = metric.compute()
    print(f"epoch {epoch}, BLEU score: {results['score']:.2f}")

    # Save and upload
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
    if accelerator.is_main_process:
        tokenizer.save_pretrained(output_dir)
        # repo.push_to_hub(
        #     commit_message=f"Training in progress epoch {epoch}", blocking=False
        # )