import os
os.environ["CUDA_VISIBLE_DEVICES"]="2"
# Datset creation on DPO style

import pandas as pd
from datasets import load_dataset 

new_dataset = load_dataset('csv', data_files= {"train":'DPO_train_val.csv',"test":'DPO_test.csv'})

print(new_dataset)


def insert_prompts(example):
    example['prompt']=example['HS_INT_EMO_TAR']
    example['chosen']=example['Ground_Counter Speech']
    example['rejected']=example['Gen_Counter Speech']

    return example

dataset = new_dataset.map(insert_prompts,batched=True)

print(dataset)

from trl import ORPOTrainer, ORPOConfig

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, TrainingArguments, Seq2SeqTrainingArguments,default_data_collator,get_linear_schedule_with_warmup,AdamW
import torch
import pandas as pd
import numpy as np
from peft import get_peft_config, get_peft_model, get_peft_model_state_dict, PrefixTuningConfig, TaskType

from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from peft import PeftModel, PeftConfig



peft_model_id_parent = "Model_checkpoints/HIPO_phase_2_FLAN_xxl/final_epoch"
config_parent = PeftConfig.from_pretrained(peft_model_id_parent)
policy_model = AutoModelForSeq2SeqLM.from_pretrained(config_parent.base_model_name_or_path,
                                                     #device_map={"":1},
                                                     torch_dtype=torch.bfloat16)
policy_model = PeftModel.from_pretrained(policy_model, peft_model_id_parent)
tokenizer = AutoTokenizer.from_pretrained(config_parent.base_model_name_or_path)

print(next(policy_model.parameters()).device)

orpo_config = ORPOConfig(
    beta=0.1,
    max_length=512,
    max_prompt_length=512,
    is_encoder_decoder=True,
    output_dir="./ORPO_results",
    evaluation_strategy="steps",
    do_eval=True,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=4,
    log_level="debug",
    save_steps=10,
    logging_steps=10,
    learning_rate=1e-4,
    eval_steps=10,
    num_train_epochs=50,
    max_steps=5000,
    warmup_steps=20,
    lr_scheduler_type="linear",
    remove_unused_columns=True,
    #local_rank=1,
    load_best_model_at_end=True,
    save_strategy = "steps",
)

trainer = ORPOTrainer( 
    model=policy_model,
    #ref_model=ref_model,
    args = orpo_config,
    train_dataset= dataset['train'],
    eval_dataset=dataset['test'],
    tokenizer = tokenizer,
)

trainer.train()
trainer.save_model("final_checkpoint_ORPO")
