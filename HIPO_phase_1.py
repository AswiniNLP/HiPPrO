from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig,T5ForConditionalGeneration,T5Tokenizer, default_data_collator,get_linear_schedule_with_warmup,AdamW
import torch
import time
import evaluate
import pandas as pd
import numpy as np

from peft import get_peft_config, get_peft_model, get_peft_model_state_dict, PrefixTuningConfig, TaskType
#from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

device = "cuda:3"


from datasets import load_dataset

# Kindly chnage the model path here
model_name = '/home/models/google-flan-t5-xxl'
tokenizer = AutoTokenizer.from_pretrained(model_name)

peft_config = PrefixTuningConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    inference_mode=False,
    num_virtual_tokens=3
    )




new_dataset = load_dataset('csv', data_files= {"train":'train_val.csv',"test":'test.csv'})

print(new_dataset)
# defining prompt based input setting
def tokenize_function(example):


    example['input_ids'] = tokenizer(example['HS_INT_TAR'],
                                   padding='max_length',
                                   max_length = 512,
                                   truncation = True,
                                   return_tensors='pt').input_ids
    example['attention_mask'] = tokenizer(example['HS_INT_TAR'],
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

model = AutoModelForSeq2SeqLM.from_pretrained(model_name,torch_dtype=torch.bfloat16)

# Initialize the model and then pass it through the get_peft_model
model = get_peft_model(model, peft_config)

# Now this will give the complete form of the PEFT based soft prompt model and which is ready for training
print(model.print_trainable_parameters())


optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=(len(train_dataloader) * num_epochs),
)



model = model.to(device)
eval_epoch_loss_list = [10000000]
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for step, batch in enumerate(tqdm(train_dataloader)):
        batch = {k: v.to(device) for k, v in batch.items()}
        #print(batch['input_ids'].shape)
        outputs = model(**batch)
        loss = outputs.loss
        total_loss += loss.detach().float()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    model.eval()
    eval_loss = 0
    eval_preds = []
    for step, batch in enumerate(tqdm(eval_dataloader)):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        loss = outputs.loss
        eval_loss += loss.detach().float()
        eval_preds.extend(
            tokenizer.batch_decode(torch.argmax(outputs.logits, -1).detach().cpu().numpy(), skip_special_tokens=True)
        )

    eval_epoch_loss = eval_loss / len(eval_dataloader)
    eval_ppl = torch.exp(eval_epoch_loss)
    train_epoch_loss = total_loss / len(train_dataloader)
    train_ppl = torch.exp(train_epoch_loss)
    print(f"{epoch=}: {train_ppl=} {train_epoch_loss=} {eval_ppl=} {eval_epoch_loss=}")
    for i in range(10):
        print(eval_preds[i])
        print('................................')

    if min(eval_epoch_loss_list)>eval_epoch_loss:
        saved_epoch = epoch
        eval_epoch_loss_list.append(eval_epoch_loss)
        print(f'Saving the model checkpoint during the epoch {saved_epoch} with epoch loss {eval_epoch_loss}')
        model.save_pretrained(f"Model_checkpoints/HIPO_phase_1_FLAN_xxl/final_epoch")
    else:
        print(f'Already saved the model checkpoint during the epoch {saved_epoch} with epoch loss {min(eval_epoch_loss_list)}')
        eval_epoch_loss_list.append(eval_epoch_loss)
    
        
