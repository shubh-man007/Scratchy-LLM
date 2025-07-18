from datasets import Dataset, DatasetDict, load_dataset, interleave_datasets, load_from_disk

def get_SQL_dataset():
    try:
        dataset = load_from_disk("merged_dataset")
        return dataset
    except:
        dataset_scc_train = load_dataset("b-mc2/sql-create-context", split='train[:80%]')
        dataset_scc_test  = load_dataset("b-mc2/sql-create-context", split='train[-20%:-10%]')
        dataset_scc_val   = load_dataset("b-mc2/sql-create-context", split='train[-10%:]')

        dataset_tts_train = load_dataset("Clinton/Text-to-sql-v1", split='train[:80%]')
        dataset_tts_train = dataset_tts_train.remove_columns(['source', 'text'])
        dataset_tts_train = dataset_tts_train.rename_columns({'instruction': 'question', 'input': 'context', 'response': 'answer'})
        dataset_tts_test  = load_dataset("Clinton/Text-to-sql-v1", split='train[-20%:-10%]')
        dataset_tts_test  = dataset_tts_test.remove_columns(['source', 'text'])
        dataset_tts_test  = dataset_tts_test.rename_columns({'instruction': 'question', 'input': 'context', 'response': 'answer'})
        dataset_tts_val   = load_dataset("Clinton/Text-to-sql-v1", split='train[-10%:]')
        dataset_tts_val   = dataset_tts_val.remove_columns(['source', 'text'])
        dataset_tts_val   = dataset_tts_val.rename_columns({'instruction': 'question', 'input': 'context', 'response': 'answer'})

        dataset_ks_train  = load_dataset("knowrohit07/know_sql", split='validation[:80%]')
        dataset_ks_test   = load_dataset("knowrohit07/know_sql", split='validation[-20%:-10%]')
        dataset_ks_val    = load_dataset("knowrohit07/know_sql", split='validation[-10%:]')

        dataset = DatasetDict({ 'train': interleave_datasets([dataset_scc_train, dataset_tts_train, dataset_ks_train]),
                                'test': interleave_datasets([dataset_scc_test, dataset_tts_test, dataset_ks_test]),
                                'validation': interleave_datasets([dataset_scc_val, dataset_tts_val, dataset_ks_val])})

        dataset.save_to_disk("merged_dataset")
        return dataset
    

def tokenize_function(example, tokenizer):
    start_prompt = "Tables:\n"
    middle_prompt = "\n\nQuestion:\n"
    end_prompt = "\n\nAnswer:\n"
  
    data_zip = zip(example['context'], example['question'])
    prompt = [start_prompt + context + middle_prompt + question + end_prompt for context, question in data_zip]
    example['input_ids'] = tokenizer(prompt, padding="max_length", truncation=True, return_tensors="pt").input_ids
    example['labels'] = tokenizer(example['answer'], padding="max_length", truncation=True, return_tensors="pt").input_ids
    return example
