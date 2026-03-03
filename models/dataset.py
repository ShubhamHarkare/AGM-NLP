#DESC: This file is responsible for generating the DataLoader to train the BERT and RoBERTa file

import torch
from torch.utils.data import Dataset
from datasets import load_from_disk


class SentimentDataset(Dataset):
    '''
    This class if responsible to fetch and load the data 
    to train the pytorch models. 
    This is a DataLoader
    '''

    def __init__(self,data_path,tokenizer,max_length = 256):
        self.data = load_from_disk(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length


    def __len__(self):
        return len(self.data)
    

    def __getitem__(self, index):
        item = self.data[index]

        label = item['label']
        assert label in (0, 1), f"Unexpected label {label} at index {index} — expected 0 or 1"

        encoding = self.tokenizer(
            item['text'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        sample = {
            'input_ids':      encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label':          torch.tensor(label, dtype=torch.long),
        }

        if 'token_type_ids' in encoding:
            sample['token_type_ids'] = encoding['token_type_ids'].squeeze(0)

        return sample