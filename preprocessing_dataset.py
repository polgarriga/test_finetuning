from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer

SEED=42
DATA_FILES="data/ca-de/softcatala_clean_bilingual.csv"
CHECKPOINT = "/gpfs/projects/bsc88/huggingface/models/m2m100_418M"
PREFIX = "data/ca-de"

def dataframe_to_hf(df,src,tgt):
    translations = []
    for s,t in zip(df[src],df[tgt]):
        translations.append({src:s,tgt:t})
    return {'id':range(len(translations)), 'translation':translations}

print(f"Loading the dataset {DATA_FILES}")

ds = load_dataset("csv", data_files=DATA_FILES)
shuffled_ds = ds.shuffle(seed=SEED)

#load dataset in HuggingFace format
dataset = {}
train_hf_split = dataframe_to_hf(shuffled_ds['train'][:-5000],'de','ca')
valid_hf_split = dataframe_to_hf(shuffled_ds['train'][-5000:],'de','ca')
#print(test_hf_split)
train_hf_split = Dataset.from_dict(train_hf_split)
valid_hf_split = Dataset.from_dict(valid_hf_split)
dataset = DatasetDict({"train": train_hf_split, "valid":valid_hf_split})

tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, return_tensors="pt", src_lang="de", tgt_lang="ca")

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

print(f"Tokenizing the dataset...")

tokenized_dataset = dataset.map(preprocess_function, batched=True, num_proc=128)

print(f"Saving the tokenized dataset...")

tokenized_dataset.save_to_disk(PREFIX)
