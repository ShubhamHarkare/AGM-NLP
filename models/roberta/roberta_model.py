
#TODO: This fle will only contain the code for the roberta model
#! Make sure that you write the model definition as close as to that of models/bert/bert_model.py
#Desc: This page contains the pytorch model that helps in creating the RoBERTa model
import torch
import torch.nn as nn
from transformers import RobertaModel


class RobertaClassifier(nn.Module):
    def __init__(self,num_labels: int = 2, dropout: float = 0.1):
        super(RobertaClassifier,self).__init__()
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.roberta.config.hidden_size,num_labels)


    def forward(self,input_ids,attention_mask,token_type_ids = None):
        outputs = self.roberta(
            input_ids = input_ids, attention_mask = attention_mask, token_type_ids = token_type_ids
        )
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return logits
