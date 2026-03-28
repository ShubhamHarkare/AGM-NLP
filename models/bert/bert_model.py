#Desc: This page contains the pytorch model that helps in creating the BERT model
import torch
import torch.nn as nn
from transformers import BertModel


class BertClassifier(nn.Module):
    def __init__(self,num_labels: int = 2, dropout: float = 0.1):
        super(BertClassifier,self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size,num_labels)


    def forward(self,input_ids,attention_mask,token_type_ids = None):
        outputs = self.bert(
            input_ids = input_ids, attention_mask = attention_mask, token_type_ids = token_type_ids
        )
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return logits
