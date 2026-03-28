
#DESC: This file contains the Domain Adversial Neural Network code
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from GradientReversalLayer import GradientReversalLayer

from transformers import RobertaModel


class DANNModel(nn.Module):
    def __init__(self, num_labels: int = 2,
                 num_domain: int = 4, dropout: float = 0.1,
                 lambda_ : float = 1.0):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.dropout = nn.Dropout(dropout)
        self.num_domain = num_domain
        self.lambda_ = lambda_
        self.sentiment_classifier = nn.Linear(768,num_labels)
        self.grl = GradientReversalLayer(lambda_=lambda_)
        self.domain_classifier = nn.Linear(768,num_domain)

    def update_lambda(self,current_step,total_steps):
        #! This function is important as we are going to change the value of lambda_ as the training progresses. We need to have good lambda value which is low at the start and then keeps going up

        progress = current_step / total_steps
        new_lambda =  2 / (1 + math.exp(-10 * progress)) - 1
        self.grl.lambda_ = new_lambda

    def forward(self,input_ids,attention_mask):
        outputs = self.roberta(
            input_ids = input_ids, attention_mask = attention_mask
        )
        pooled_output = self.dropout(outputs.pooler_output)

        sentiment_logits = self.sentiment_classifier(pooled_output)
        domain_logits = self.domain_classifier(self.grl(pooled_output))

        return sentiment_logits,domain_logits

        





    
        
