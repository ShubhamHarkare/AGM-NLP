
#DESC: this file is responsilble for generating the AGM (Attribution Guided Masking) model

import torch
import torch.nn as nn

from transformers import RobertaModel

class AGMModel(nn.Module):
    def __init__(self, num_labels:int = 2, dropout:float = 0.1) -> None:
        super().__init__()
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.roberta.config.hidden_size,num_labels)


    def forward(self,input_ids,attention_mask):
        outputs = self.roberta(input_ids,attention_mask)

        pooled_output = outputs.pooler_output
        last_hidden_state = outputs.last_hidden_state
        logits = self.classifier(pooled_output)


        return logits,pooled_output,last_hidden_state




def compute_gradient_input(last_hidden_state):
    assert last_hidden_state.grad is not None, \
        "No gradient found — did you call retain_grad() and loss.backward()?"
    grads = last_hidden_state.grad
    attribution = grads * last_hidden_state
    return attribution.sum(dim=2)                         # [batch, seq_len]



def detect_spurious_tokens(attribution,tau_high = 0.75):
    threshold = torch.quantile(attribution,tau_high,dim = 1,keepdim=True)

    spurious_mask = attribution > threshold

    return spurious_mask


def generate_counterfactual(input_ids, spurious_mask, mlm_model, mask_token_id):
    '''
    Generates counterfactual input by replacing spurious tokens
    with MLM predictions.
    
    Args:
        input_ids:      [batch, seq_len] — original token IDs
        spurious_mask:  [batch, seq_len] — True where token is spurious
        mlm_model:      RobertaForMaskedLM — for generating replacements
        mask_token_id:  int — RoBERTa's [MASK] token ID
    
    Returns:
        counterfactual_ids: [batch, seq_len] — input_ids with spurious 
                            tokens replaced by MLM predictions
    '''

    # Step 1: clone input_ids — never modify original in place
    counterfactual_ids = input_ids.clone()

    # Step 2: replace spurious token positions with [MASK]
    counterfactual_ids[spurious_mask] = mask_token_id

    # Step 3: run MLM model to get predictions
    # no gradient needed — counterfactual generation is not differentiated
    with torch.no_grad():
        mlm_outputs = mlm_model(
            input_ids=counterfactual_ids,
            attention_mask=torch.ones_like(counterfactual_ids)
        )
        # mlm_outputs.logits shape: [batch, seq_len, vocab_size]
        mlm_logits = mlm_outputs.logits

    # Step 4: get predicted replacement tokens via argmax
    # shape: [batch, seq_len]
    predicted_ids = torch.argmax(mlm_logits, dim=2)

    # Step 5: only replace the masked positions with predictions
    # non-spurious tokens keep their original IDs
    counterfactual_ids = input_ids.clone()
    counterfactual_ids[spurious_mask] = predicted_ids[spurious_mask]

    return counterfactual_ids


def filter_counterfactual(input_ids,counterfactual_ids,model,attention_mask):
    with torch.no_grad():
        logits,_,_ = model(input_ids,attention_mask)
        original_labels = torch.argmax(logits,dim = 1)

        counterfactual_logits,_,_ = model(counterfactual_ids,attention_mask)

        counterfactual_labels = torch.argmax(counterfactual_logits,dim = 1)



    return counterfactual_labels == original_labels