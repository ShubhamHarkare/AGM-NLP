
#DESC: This file is responsible for calculating the mean Integrated Gradients value for all the RoBERTa models that we have trained so far
from captum.attr import LayerIntegratedGradients
from sklearn.metrics.pairwise import cosine_similarity
import torch


def compute_mean_ig(model,dataloader,device):
    '''
    Computes the mean Integrated Gradients attribution vector
    for a given model over a dataset

    Returns a tensor of shape [MAX_LENGTH] - one scalar attribution per token position
    averaged across all samples in the dataloader
    '''

    #TODO: Setting model to eval - no dropout and no gradient updates
    model.eval()

    #TODO: Using captum to set the forward function
    def __forward_func(input_ids,attention_mask):
        return model(input_ids,attention_mask)
    

    lig = LayerIntegratedGradients(
        __forward_func,
        model.roberta.embeddings.word_embeddings
    )

    #TODO: Accumulating attribution vectors across all batches

    all_attributions = []
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)


        with torch.no_grad():
            logits = model(input_ids,attention_mask)

        target = torch.argmax(logits,dim = 1)

        #TODO: Creating the baseline which represents abscence of information
        baseline = torch.zeros_like(input_ids)


        #! Computing the IG attribution here
        attributions = lig.attribute(
            inputs=input_ids,
            baselines=baseline,
            additional_forward_args=(attention_mask,),
            target=target,
            n_steps=20
        )

        attributions = attributions.sum(dim = 2)


        all_attributions.append(attributions.detach().cpu())
        del attributions
        torch.cuda.empty_cache()


    all_attributions = torch.cat(all_attributions,dim = 0)

    mean_ig = all_attributions.mean(dim = 0)
    return mean_ig




def compute_ads(mean_ig_score,mean_ig_taget):
    '''
    Computes Attribution Drift Score between source and target domains.
    
    ADS(S,T) = 1 - cosine_similarity(meanIG_S, meanIG_T)
    
    Higher ADS = more attribution drift = expect larger generalization gap
    '''

    source = mean_ig_score.numpy().reshape(1,-1)
    target = mean_ig_taget.numpy().reshape(1,-1)


    cos_sim = cosine_similarity(source,target)[0][0]

    ads = 1 - cos_sim

    return float(ads)
