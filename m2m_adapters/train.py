from datasets import load_dataset, Dataset, DatasetDict, load_from_disk
from modeling_m2m_100_parallel import M2M100AdapterForConditionalGeneration
from configuration_m2m_100 import M2M100AdapterConfig
from transformers import AutoTokenizer, AutoModel, DataCollatorForSeq2Seq
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import AutoConfig, AutoModel
import numpy as np
import os

import torch


CHECKPOINT =  "facebook/m2m100_418M"
DATASET_LOC = "data/ca-de"
source_lang = "de"
target_lang = "ca"

def freeze_params(model):
    #Freeze all parameters not in the adapters
    trainable_parameters = []
    for name, param in model.named_parameters():
        if not "adapter" in name:
            param.requires_grad = False
        else:
            print('Trainable param:', name)
            trainable_parameters.append(param)
    params = sum([np.prod(p.size()) for p in trainable_parameters])
    print('Total number of trainable params:', params)
    return model


def dataframe_to_hf(df,src,tgt):
    translations = []
    for s,t in zip(df[src],df[tgt]):
        translations.append({src:s,tgt:t})
    return {'id':range(len(translations)), 'translation':translations}



def print_mismatched_state_dict_keys(model_state_dict, pretrained_state_dict):
    missing = 0
    for k in model_state_dict.keys():
        if k not in pretrained_state_dict:
            missing += 1
            print('Missing k', k)

    print("Total missing keys", missing)

    unexpected = 0
    for k in pretrained_state_dict.keys():
        if k not in model_state_dict:
            unexpected += 1
            print('Unexpected k', k)
    print("Total unexpected keys", unexpected)


def load_pretrained_state_dict(filename):
    pretrained_state_dict =  torch.load(filename)
    #return {k.replace('model.',''):v for k,v in pretrained_state_dict.items()}
    return pretrained_state_dict


AutoConfig.register("M2M100_Adapter", M2M100AdapterConfig)
AutoModelForSeq2SeqLM.register(M2M100AdapterConfig, M2M100AdapterForConditionalGeneration)

config = M2M100AdapterConfig()
model = AutoModelForSeq2SeqLM.from_config(config) 

print("Model created", len(model.state_dict()))




print("Loading tokenizer")

tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, 
                                            return_tensors="pt", 
                                            src_lang=source_lang, 
                                            tgt_lang=target_lang)


print(f"Loading the dataset {DATASET_LOC}")

tokenized_dataset = load_from_disk(DATASET_LOC)


print("Loading pretrained")
pretrained_state_dict = load_pretrained_state_dict(os.path.join(CHECKPOINT, 'pytorch_model.bin')) # change
print_mismatched_state_dict_keys(model.state_dict(),pretrained_state_dict)


print("Loading pretrained weights")
model.load_state_dict(pretrained_state_dict,strict=False)
model = freeze_params(model)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

training_args = Seq2SeqTrainingArguments(
        output_dir="./results/parallel-adapter",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        weight_decay=0.01,
        save_total_limit=1,
        num_train_epochs=1,
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

print("Starting training...")

trainer.train()
