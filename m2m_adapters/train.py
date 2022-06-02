from modeling_m2m_100_parallel import M2M100Model
from configuration_m2m_100 import M2M100Config
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch


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
    return {k.replace('model.',''):v for k,v in pretrained_state_dict.items()}

tokenizer = AutoTokenizer.from_pretrained("facebook/m2m100_418M") # change
print("Tokenizer loaded")

config = M2M100Config()
model = M2M100Model(config)
print("Model created", len(model.state_dict()))

#trained_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/m2m100_418M")
pretrained_state_dict = load_pretrained_state_dict('/home/bscuser/Documents/m2m100_418M/pytorch_model.bin') # change
print("Pretrained model loaded", len(pretrained_state_dict))
print_mismatched_state_dict_keys(model.state_dict(),pretrained_state_dict)

model.load_state_dict(pretrained_state_dict,strict=False)
print("Pretrained weights loaded")