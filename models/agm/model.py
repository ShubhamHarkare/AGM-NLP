import torch
import torch.nn as nn
from transformers import RobertaModel

class AGMModel(nn.Module):
    def __init__(self, num_labels:int = 2, dropout:float = 0.1) -> None:
        super().__init__()
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.roberta.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids, attention_mask)
        pooled_output = outputs.pooler_output
        last_hidden_state = outputs.last_hidden_state
        
        # FIX: Added dropout before classifier
        logits = self.classifier(self.dropout(pooled_output))

        return logits, pooled_output, last_hidden_state


# FIX: Accept grads as an argument for double-backprop
def compute_gradient_input(last_hidden_state, grads):
    attribution = grads * last_hidden_state
    return attribution.sum(dim=2)  # [batch, seq_len]


def detect_spurious_tokens(attribution, tau_high=0.75):
    # FIX: Use absolute magnitude to catch highly influential negative tokens
    abs_attribution = torch.abs(attribution)
    threshold = torch.quantile(abs_attribution, tau_high, dim=1, keepdim=True)
    spurious_mask = abs_attribution > threshold
    return spurious_mask


# FIX: Added attention_mask to arguments
def generate_counterfactual(input_ids, attention_mask, spurious_mask, mlm_model, mask_token_id):
    # Step 1: clone input_ids
    counterfactual_ids = input_ids.clone()

    # Step 2: replace spurious token positions with [MASK]
    counterfactual_ids[spurious_mask] = mask_token_id

    # Step 3: run MLM model to get predictions
    with torch.no_grad():
        mlm_outputs = mlm_model(
            input_ids=counterfactual_ids,
            attention_mask=attention_mask # FIX: Use actual attention mask, not ones
        )
        mlm_logits = mlm_outputs.logits

    # Step 4: get predicted replacement tokens via argmax
    predicted_ids = torch.argmax(mlm_logits, dim=2)

    # Step 5: replace ONLY the masked positions
    counterfactual_ids = input_ids.clone()
    counterfactual_ids[spurious_mask] = predicted_ids[spurious_mask]

    return counterfactual_ids


def filter_counterfactual(input_ids, counterfactual_ids, model, attention_mask):
    with torch.no_grad():
        logits, _, _ = model(input_ids, attention_mask)
        original_labels = torch.argmax(logits, dim=1)

        counterfactual_logits, _, _ = model(counterfactual_ids, attention_mask)
        counterfactual_labels = torch.argmax(counterfactual_logits, dim=1)

    return counterfactual_labels == original_labels