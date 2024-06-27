
import torch
import torch.nn.functional as F
from transformers.generation import TopKLogitsWarper, TopPLogitsWarper
from ..utils.infer_utils import CustomRepetitionPenaltyLogitsProcessorRepeat

def infer_code(
    models,
    text, 
    spk_emb = None,
    top_P = 0.7, 
    top_K = 20, 
    temperature = 0.3, 
    repetition_penalty = 1.05,
    max_new_token = 2048,
    **kwargs
):
    
    device = next(models['vocos'].parameters()).device
    
    if not isinstance(text, list): 
        text = [text]
        
    # if not isinstance(temperature, list):
    #     temperature = [temperature] * models['gpt'].num_vq
    
    if spk_emb is not None:
        text = [f'[Stts][spk_emb]{i}[Ptts]' for i in text] 
    else:
        text = [f'[Stts][empty_spk]{i}[Ptts]' for i in text]
    
    text_token = models['tokenizer'](text, return_tensors='pt', add_special_tokens=False, padding=True).to(device) 

    print(text_token)
    input_ids = text_token['input_ids'][...,None].expand(-1, -1, models['gpt'].num_vq) # torch.Size([1, 66, 4])
    
    inputs = {
        'input_ids': input_ids,
        'attention_mask': text_token['attention_mask'], # 1 1 ... 1
    }

    if spk_emb is not None: # 若指定了speaker embedding，那把spk_emb 21143换成指定的
        spk_emb = F.normalize(spk_emb.to(device)[None].expand(len(text), -1), p=2.0, dim=1, eps=1e-12)  
    
    num_code = 625 # models['gpt'].emb_code[0].num_embeddings - 1 # 626-1 # ModuleList((0-3): 4 x Embedding(626, 768))
    
    LogitsWarpers = []
    if top_P is not None:
        LogitsWarpers.append(TopPLogitsWarper(top_P, min_tokens_to_keep=3))
    if top_K is not None:
        LogitsWarpers.append(TopKLogitsWarper(top_K, min_tokens_to_keep=3))
        
    LogitsProcessors = []
    if repetition_penalty is not None and repetition_penalty != 1:
        LogitsProcessors.append(CustomRepetitionPenaltyLogitsProcessorRepeat(repetition_penalty, num_code, 16))
    
    result = models['gpt'].generate_code(
        inputs['input_ids'], 
        spk_emb = spk_emb, # replace 21143
        temperature = torch.tensor(temperature, device=device), 
        attention_mask = inputs['attention_mask'],
        LogitsWarpers = LogitsWarpers,
        LogitsProcessors = LogitsProcessors,
        eos_token = num_code, 
        max_new_token = max_new_token,
        **kwargs
    )
    
    return result


def refine_text(
    models, 
    text,
    top_P = 0.7, 
    top_K = 20, 
    temperature = 0.7, 
    repetition_penalty = 1.0,
    max_new_token = 384,
    prompt = '',
    **kwargs
):
    
    device = next(models['vocos'].parameters()).device
    
    if not isinstance(text, list): 
        text = [text]
    
    assert len(text), 'text should not be empty'

    text = [f"[Sbreak]{i}[Pbreak]{prompt}" for i in text]
    text_token = models['tokenizer'](text, return_tensors='pt', add_special_tokens=False, padding=True).to(device)
    text_mask = torch.ones(text_token['input_ids'].shape, dtype=bool, device=device)

    inputs = {
        'input_ids': text_token['input_ids'][...,None].expand(-1, -1, models['gpt'].num_vq),
        'text_mask': text_mask,
        'attention_mask': text_token['attention_mask'],
    }
    print("text_token origin", text_token)
    LogitsWarpers = []
    if top_P is not None:
        LogitsWarpers.append(TopPLogitsWarper(top_P, min_tokens_to_keep=3))
    if top_K is not None:
        LogitsWarpers.append(TopKLogitsWarper(top_K, min_tokens_to_keep=3))
        
    LogitsProcessors = []
    if repetition_penalty is not None and repetition_penalty != 1:
        LogitsProcessors.append(CustomRepetitionPenaltyLogitsProcessorRepeat(repetition_penalty, len(models['tokenizer']), 16))
    
    result = models['gpt'].generate_text(
        inputs['input_ids'],
        temperature = torch.tensor([temperature,], device=device), 
        attention_mask = inputs['attention_mask'],
        LogitsWarpers = LogitsWarpers,
        LogitsProcessors = LogitsProcessors,
        eos_token = torch.tensor(models['tokenizer'].convert_tokens_to_ids('[Ebreak]'), device=device)[None], # 21136
        max_new_token = max_new_token, 
        **kwargs
    )
    return result