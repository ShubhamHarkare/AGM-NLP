from captum.attr import LayerIntegratedGradients
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import torch
import numpy as np

def compute_token_ig(model, dataloader, device):
    '''
    Computes mean IG attribution per TOKEN ID (not position).
    Returns a dictionary: {token_id: mean_attribution}
    This enables directional, shared-vocabulary ADS computation.
    '''
    model.eval()

    def forward_func(input_ids, attention_mask):
        return model(input_ids, attention_mask)

    lig = LayerIntegratedGradients(
        forward_func,
        model.roberta.embeddings.word_embeddings
    )

    # accumulate attributions per token ID
    token_attributions = defaultdict(list)

    for batch in dataloader:
        input_ids      = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        with torch.no_grad():
            logits = model(input_ids, attention_mask)
        target = torch.argmax(logits, dim=1)

        baseline = torch.zeros_like(input_ids)

        attributions = lig.attribute(
            inputs=input_ids,
            baselines=baseline,
            additional_forward_args=(attention_mask,),
            target=target,
            n_steps=20
        )

        # sum across embedding dim → [batch, seq_len]
        attributions = attributions.sum(dim=2).detach().cpu()
        input_ids_cpu = input_ids.cpu()

        # map each attribution to its token ID
        for sample_idx in range(input_ids_cpu.shape[0]):
            for pos in range(input_ids_cpu.shape[1]):
                token_id = input_ids_cpu[sample_idx, pos].item()
                attr_val = attributions[sample_idx, pos].item()
                # skip padding token (id=1 in RoBERTa)
                if token_id != 1:
                    token_attributions[token_id].append(attr_val)

        del attributions
        torch.cuda.empty_cache()

    # compute mean attribution per token ID
    mean_token_ig = {
        token_id: np.mean(attrs)
        for token_id, attrs in token_attributions.items()
    }

    return mean_token_ig


def compute_ads_shared_vocab(token_ig_source, token_ig_target, min_attribution=1e-5):
    '''
    Computes directional ADS using shared vocabulary only.
    Filters out tokens with near-zero attribution in both domains.
    
    Args:
        token_ig_source: dict {token_id: mean_attribution} from source model on source data
        token_ig_target: dict {token_id: mean_attribution} from source model on target data
        min_attribution: minimum attribution threshold to include token
    
    Returns:
        float: ADS score
        int: number of shared tokens used
    '''
    # find shared vocabulary
    shared_tokens = set(token_ig_source.keys()) & set(token_ig_target.keys())

    # filter by attribution magnitude — removes stop words naturally
    shared_tokens = {
        t for t in shared_tokens
        if abs(token_ig_source[t]) > min_attribution
        or abs(token_ig_target[t]) > min_attribution
    }

    if len(shared_tokens) < 10:
        print(f"Warning: only {len(shared_tokens)} shared tokens — ADS may be unreliable")
        return 0.0, len(shared_tokens)

    # build aligned vectors
    source_vec = np.array([token_ig_source[t] for t in shared_tokens])
    target_vec = np.array([token_ig_target[t] for t in shared_tokens])

    # compute cosine similarity
    cos_sim = cosine_similarity(
        source_vec.reshape(1, -1),
        target_vec.reshape(1, -1)
    )[0][0]

    ads = 1 - cos_sim
    return float(ads), len(shared_tokens)