from transformers import GPT2Tokenizer, GPT2LMHeadModel, AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig, default_data_collator
import torch
from peft import PeftModel, PeftConfig
from datasets import load_dataset
from tqdm import tqdm
import pandas as pd
from torch.utils.data import DataLoader

device = 'cuda:2'

df = pd.read_csv("test.csv")
###################################################
peft_model_id_parent = "please paste here the peft model path"
config_parent = PeftConfig.from_pretrained(peft_model_id_parent)
model_parent = AutoModelForSeq2SeqLM.from_pretrained(config_parent.base_model_name_or_path,torch_dtype=torch.bfloat16)
model_parent = PeftModel.from_pretrained(model_parent, peft_model_id_parent)
tokenizer = AutoTokenizer.from_pretrained(config_parent.base_model_name_or_path)
##################################################

dataset_original = load_dataset('csv',
                                data_files={'test': "test.csv"})

print(len(dataset_original['test']))


model_parent = model_parent.to(device)

for i in tqdm(range(len(dataset_original['test']))):
    
    input_ids = tokenizer(dataset_original['test']['HS_INT_EMO_TAR'][i], return_tensors='pt').input_ids

    input_ids=input_ids.to(device)
    original_model_outputs = model_parent.generate(input_ids=input_ids,max_new_tokens=512,
                                                                                      )
    original_model_test_output = tokenizer.decode(original_model_outputs[0].detach().cpu().numpy(), skip_special_tokens=True)
    
    #print(original_model_test_output)
    eval_preds.append(original_model_test_output)
    eval_hate.append(dataset_original['test']['hatespeech'][i])
    eval_CS.append(dataset_original['test']['counterspeech'][i])
    
    
    print("Hate Speech :")
    print(".............................")
    print(eval_hate[i])
    print("Counter Speech :")
    print(".............................")
    print(eval_preds[i])
    
df['responses'] = eval_preds   

df.to_csv('HiPPro_VT3.csv', index=False)
