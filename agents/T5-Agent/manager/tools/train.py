from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict, load_dataset, interleave_datasets, load_from_disk
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig, TrainingArguments, Trainer
from utils import get_SQL_dataset, tokenize_function
import torch
import time
import evaluate
import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings("ignore")


#<------Initialize Model Weights and Tokenizer------>
model_name='t5-small'
tokenizer = AutoTokenizer.from_pretrained(model_name)
original_model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
original_model = original_model.to('cuda')


#<------Obtain Dataset and Tokenize------>
dataset = get_SQL_dataset()

try:
    tokenized_datasets = load_from_disk("tokenized_datasets")
except:
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(['question', 'context', 'answer'])
    
    tokenized_datasets.save_to_disk("tokenized_datasets")

print(f"Shapes of the datasets:")
print(f"Training: {tokenized_datasets['train'].shape}")
print(f"Validation: {tokenized_datasets['validation'].shape}")
print(f"Test: {tokenized_datasets['test'].shape}")
print(tokenized_datasets)


#<------Train Model------>
try:
    finetuned_model = AutoModelForSeq2SeqLM.from_pretrained("finetuned_model_2_epoch")
    finetuned_model = finetuned_model.to('cuda')
    to_train = False

except:
    to_train = True
    finetuned_model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    finetuned_model = finetuned_model.to('cuda')
    tokenizer = AutoTokenizer.from_pretrained(model_name)

if to_train:
    output_dir = f'./sql-training-{str(int(time.time()))}'

    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=5e-3,
        num_train_epochs=2,
        per_device_train_batch_size=16,     # batch size per device during training
        per_device_eval_batch_size=16,      # batch size for evaluation
        weight_decay=0.01,
        logging_steps=50,
        evaluation_strategy='steps',        # evaluation strategy to adopt during training
        eval_steps=500,                     # number of steps between evaluation
    )

    trainer = Trainer(
        model=finetuned_model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['validation'],
    )
    
    trainer.train()
    
    finetuned_model.save_pretrained("finetuned_model_2_epoch")

finetuned_model = AutoModelForSeq2SeqLM.from_pretrained("finetuned_model_2_epoch")
finetuned_model = finetuned_model.to('cuda')

# #<------Test Model------->
# questions = dataset['test'][0:25]['question']
# contexts = dataset['test'][0:25]['context']
# human_baseline_answers = dataset['test'][0:25]['answer']

# original_model_answers = []
# finetuned_model_answers = []

# for idx, question in enumerate(questions):
    
#     prompt = f"""Tables:
# {contexts[idx]}

# Question:
# {question}

# Answer:
# """
      
#     input_ids = tokenizer(prompt, return_tensors="pt").input_ids
#     input_ids = input_ids.to('cuda')

#     human_baseline_text_output = human_baseline_answers[idx]
    
#     original_model_outputs = original_model.generate(input_ids=input_ids, generation_config=GenerationConfig(max_new_tokens=300))
#     original_model_text_output = tokenizer.decode(original_model_outputs[0], skip_special_tokens=True)
#     original_model_answers.append(original_model_text_output)
    
#     finetuned_model_outputs = finetuned_model.generate(input_ids=input_ids, generation_config=GenerationConfig(max_new_tokens=300))
#     finetuned_model_text_output = tokenizer.decode(finetuned_model_outputs[0], skip_special_tokens=True)
#     finetuned_model_answers.append(finetuned_model_text_output)

# zipped_summaries = list(zip(human_baseline_answers, original_model_answers, finetuned_model_answers))
 
# df = pd.DataFrame(zipped_summaries, columns = ['human_baseline_answers', 'original_model_answers', 'finetuned_model_answers'])
# # print(df)

# #<------ROGUE Score------>
# rouge = evaluate.load('rouge')

# original_model_results = rouge.compute(
#     predictions=original_model_answers,
#     references=human_baseline_answers[0:len(original_model_answers)],
#     use_aggregator=True,
#     use_stemmer=True,
# )
# print('ORIGINAL MODEL:')
# print(original_model_results)

# print('-'*50)

# finetuned_model_results = rouge.compute(
#     predictions=finetuned_model_answers,
#     references=human_baseline_answers[0:len(finetuned_model_answers)],
#     use_aggregator=True,
#     use_stemmer=True,
# )
# print('FINE-TUNED MODEL:')
# print(finetuned_model_results)

# # Output :
# # ORIGINAL MODEL:
# # {'rouge1': 0.03210393397380822, 'rouge2': 0.005, 'rougeL': 0.030087045570916536, 'rougeLsum': 0.03115103231013564}
# # FINE-TUNED MODEL:
# # {'rouge1': 0.9449139418171086, 'rouge2': 0.9166952164685527, 'rougeL': 0.9364083176806925, 'rougeLsum': 0.9360694621851381}
# # ​
