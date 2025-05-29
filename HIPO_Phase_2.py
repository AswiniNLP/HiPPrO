from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, default_data_collator,get_linear_schedule_with_warmup,AdamW
from transformers import GenerationConfig,T5ForConditionalGeneration,T5Tokenizer
import torch
import time
#import evaluate
import pandas as pd
import numpy as np

from peft import get_peft_config, get_peft_model, get_peft_model_state_dict, PrefixTuningConfig, TaskType
#from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import os
from peft import PeftModel, PeftConfig

os.environ["TOKENIZERS_PARALLELISM"] = "false"
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

device = "cuda:1"


from datasets import load_dataset

#########################################
# Kindly chnage the model path here
model_name = '/home/models/google-flan-t5-xxl'
tokenizer = AutoTokenizer.from_pretrained(model_name)

peft_config_original = PrefixTuningConfig(
    peft_type="PREFIX_TUNING",
    inference_mode=False,
    task_type=TaskType.SEQ_2_SEQ_LM,
    num_virtual_tokens=3,
)

model_parent = AutoModelForSeq2SeqLM.from_pretrained(model_name,torch_dtype=torch.bfloat16)

# Initialize the model and then pass it through the get_peft_model
model_parent = get_peft_model(model_parent, peft_config_original)

# Now this will give the complete form of the PEFT based soft prompt model and which is ready for training
print(model_parent.print_trainable_parameters())

#################################################

peft_model_id_1 = "Model_checkpoints/HIPO_phase_1_FLAN_xxl/final_epoch"
config_1 = PeftConfig.from_pretrained(peft_model_id_1)
model_child_1 = AutoModelForSeq2SeqLM.from_pretrained(config_1.base_model_name_or_path)
model_child_1 = PeftModel.from_pretrained(model_child_1, peft_model_id_1)
#tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

##################################################



##################################################

embedding_layer_parent = model_parent.prompt_encoder.default.embedding.weight
embedding_layer_child_1 = model_child_1.prompt_encoder.default.embedding.weight
embedding_layer_parent.data[3:,:]=embedding_layer_child_1.data

#####################################################




new_dataset = load_dataset('csv', data_files= {"train":'train_val.csv',"test":'test.csv'})

print(new_dataset)

# defining prompt based input setting
def tokenize_function(example):


    example['input_ids'] = tokenizer(example['HS_INT_EMO_TAR'],
                                   padding='max_length',
                                   max_length = 512,
                                   truncation = True,
                                   return_tensors='pt').input_ids
    example['attention_mask'] = tokenizer(example['HS_INT_EMO_TAR'],
                                   padding='max_length',
                                   max_length = 512,
                                   truncation = True,
                                   return_tensors='pt').attention_mask
    example['labels'] = tokenizer(example['counterspeech'],
                                padding='max_length',
                                max_length = 512,
                                truncation = True,
                                return_tensors='pt').input_ids

    return example




# Now the example dictionary will get added by two new keys "input_ids" and 'labels'

# The dataset is actually containign three different splits: train, test split

tokenized_dataset = new_dataset.map(tokenize_function, batched=True)
#tokenized_dataset = tokenized_dataset.remove_columns(['Unnamed: 0'])

train_dataset = tokenized_dataset["train"]
eval_dataset = tokenized_dataset["test"]

train_dataset = train_dataset.remove_columns(['hatespeech', 'csType', 'counterspeech', 'hatespeechTarget','Emotion'])
eval_dataset = eval_dataset.remove_columns(['hatespeech', 'csType', 'counterspeech', 'hatespeechTarget','Emotion'])


batch_size = 4

num_epochs = 50
lr = 1e-4


train_dataloader = DataLoader(
    train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True
)
eval_dataloader = DataLoader(eval_dataset, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True)

print(len(train_dataloader))
print(len(eval_dataloader))




optimizer = torch.optim.AdamW(model_parent.parameters(), lr=lr)
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=(len(train_dataloader) * num_epochs),
)


model_parent = model_parent.to(device)
eval_epoch_loss_list = [100000]
for epoch in range(num_epochs):
    model_parent.train()
    total_loss = 0
    for step, batch in enumerate(tqdm(train_dataloader)):
        batch = {k: v.to(device) for k, v in batch.items()}
        #print(batch['input_ids'].shape)
        outputs = model_parent(**batch)
        loss = outputs.loss
        total_loss += loss.detach().float()
        loss.backward()
        optimizer.step()
        embedding_layer_parent.data[3:,:]=embedding_layer_child_1.data
        #lr_scheduler.step()
        optimizer.zero_grad()

    model_parent.eval()
    eval_loss = 0
    eval_hate = []
    eval_preds = []
    for step, batch in enumerate(tqdm(eval_dataloader)):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model_parent(**batch)
        loss = outputs.loss
        eval_loss += loss.detach().float()
        embedding_layer_parent.data[3:,:]=embedding_layer_child_1.data
        eval_preds.extend(
            tokenizer.batch_decode(torch.argmax(outputs.logits, -1).detach().cpu().numpy(), skip_special_tokens=True)
        )
        
        eval_hate.extend(
            tokenizer.batch_decode(batch['input_ids'].detach().cpu().numpy(), skip_special_tokens=True)
        )

    eval_epoch_loss = eval_loss / len(eval_dataloader)
    lr_scheduler.step(eval_epoch_loss)
    
    eval_ppl = torch.exp(eval_epoch_loss)
    train_epoch_loss = total_loss / len(train_dataloader)
    train_ppl = torch.exp(train_epoch_loss)
    print(f"{epoch=}: {train_ppl=} {train_epoch_loss=} {eval_ppl=} {eval_epoch_loss=}")
    for i in range(10):
        print('Hate speech : ')
        print('.................................')
        print(eval_hate[i])
        print('.................................')
        print('Counter Speech : ')
        print('.................................')
        print(eval_preds[i])
        print('.................................')

    

    if min(eval_epoch_loss_list)>eval_epoch_loss:
        saved_epoch = epoch
        eval_epoch_loss_list.append(eval_epoch_loss)
        print(f'Saving the model checkpoint during the epoch {saved_epoch} with epoch loss {eval_epoch_loss}')
        embedding_layer_parent.data[3:,:]=embedding_layer_child_1.data
        model_parent.save_pretrained(f"Model_checkpoints/HIPO_phase_2_FLAN_xxl/final_epoch")
    else:
        print(f'Already saved the model checkpoint during the epoch {saved_epoch} with epoch loss {min(eval_epoch_loss_list)}')
        eval_epoch_loss_list.append(eval_epoch_loss)
