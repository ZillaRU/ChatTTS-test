import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import logging
from tqdm import tqdm
from einops import rearrange
# from transformers.cache_utils import Cache

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrize as P
from torch.nn.utils.parametrizations import weight_norm
from transformers import LlamaModel, LlamaConfig
    

class GPT_warpper(nn.Module):
    def __init__(
        self, 
        gpt_config, 
        num_audio_tokens,
        num_text_tokens,
        num_vq=4,
        **kwargs,
        ):
        super().__init__()
        import llama
        self.logger = logging.getLogger(__name__)
        self.gpt = llama.TTSLlama()
        self.model_dim = self.gpt.config.hidden_size 

        self.num_vq = num_vq
        self.emb_text = nn.Embedding(num_text_tokens, self.model_dim)
        self.emb_code = nn.ModuleList([nn.Embedding(num_audio_tokens, self.model_dim) for i in range(self.num_vq)])
        self.head_text = weight_norm(nn.Linear(self.model_dim, num_text_tokens, bias=False), name='weight')
        self.head_code = nn.ModuleList([weight_norm(nn.Linear(self.model_dim, num_audio_tokens, bias=False), name='weight') for i in range(self.num_vq)])

    def build_model(self, config):
        
        configuration = LlamaConfig(**config)
        model = LlamaModel(configuration)
        print('model size: {:.3f}M'.format(sum([p.numel() for p in model.parameters()]) / 1e6))
        print(model.config)
        print(model)
        del model.embed_tokens
        
        return model


    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]
        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs
      
    def generate_code(
        self,
        inputs_ids,
        spk_emb,
        temperature, 
        eos_token, 
        attention_mask = None,
        max_new_token = 2048, 
        min_new_token = 0,
        LogitsWarpers = [],
        LogitsProcessors = [],
        return_hidden=True,
    ):
        
        with torch.no_grad():   
            hiddens = []
            
            start_idx, end_idx = inputs_ids.shape[1], torch.zeros(inputs_ids.shape[0], device=inputs_ids.device, dtype=torch.long)
            finish = torch.zeros(inputs_ids.shape[0], device=inputs_ids.device).bool()
            
            temperature = temperature[None].expand(inputs_ids.shape[0], -1)
            temperature = rearrange(temperature, "b n -> (b n) 1")

            attention_mask_cache = torch.ones((inputs_ids.shape[0], inputs_ids.shape[1]+max_new_token,), dtype=torch.bool, device=inputs_ids.device)
            if attention_mask is not None:
                attention_mask_cache[:, :attention_mask.shape[1]] = attention_mask
            
            for i in tqdm(range(max_new_token)):
                model_input = self.prepare_inputs_for_generation(inputs_ids, 
                    outputs.past_key_values if i!=0 else None, 
                    attention_mask_cache[:, :inputs_ids.shape[1]], use_cache=True)
            
                if i == 0:
                    emb = self.emb_text(model_input['input_ids'][:, :, 0])
                    if spk_emb is not None:
                        emb[inputs_ids[:, -1] == 21143] = spk_emb.to(emb.device)
                    model_input['inputs_embeds'] = emb
                else:
                    code_emb = [self.emb_code[i](model_input['input_ids'][:,:,i]) for i in range(self.num_vq)]
                    model_input['inputs_embeds'] = torch.stack(code_emb, 3).sum(3)
                
                model_input['input_ids'] = None

                outputs = self.gpt.forward(**model_input, output_attentions=False)
                
                hidden_states = outputs[0] # 🐻 [1, 57, 768]
                hiddens.append(hidden_states[:, -1])

                logits = torch.stack([self.head_code[i](hidden_states) for i in range(self.num_vq)], 3)
        
                logits = logits[:, -1].float()

                logits = rearrange(logits, "b c n -> (b n) c")
                logits_token = rearrange(inputs_ids[:, start_idx:], "b c n -> (b n) c")

                logits = logits / temperature
                
                for logitsProcessors in LogitsProcessors:
                    logits = logitsProcessors(logits_token, logits)
                    
                for logitsWarpers in LogitsWarpers:
                    logits = logitsWarpers(logits_token, logits)
                    
                if i < min_new_token:
                    logits[:, eos_token] = -torch.inf
                
                scores = F.softmax(logits, dim=-1)
            
                idx_next = torch.multinomial(scores, num_samples=1)
                
                idx_next = rearrange(idx_next, "(b n) 1 -> b n", n=self.num_vq)
                finish = finish | (idx_next == eos_token).any(1)
                inputs_ids = torch.cat([inputs_ids, idx_next.unsqueeze(1)], 1)

                end_idx = end_idx + (~finish).int()
            
                if finish.all():
                    break
            
            inputs_ids = [inputs_ids[idx, start_idx: start_idx+i] for idx, i in enumerate(end_idx.int())]
            
            hiddens = torch.stack(hiddens, 1)
            hiddens = [hiddens[idx, :i] for idx, i in enumerate(end_idx.int())]
                    
            if not finish.all():
                self.logger.warn(f'Incomplete result. hit max_new_token: {max_new_token}')    
            
            return {
                'ids': inputs_ids, # [505,4]
                'hiddens':hiddens, #[505, 768]
            }
    
    def generate_text(
        self, 
        inputs_ids,
        temperature, 
        eos_token, 
        attention_mask = None,
        max_new_token = 2048, 
        min_new_token = 0,
        LogitsWarpers = [],
        LogitsProcessors = [],
        return_hidden=False,
    ):
        self.gpt.generate_text(inputs_ids, max_new_token)
        
        with torch.no_grad():

            start_idx, end_idx = inputs_ids.shape[1], torch.zeros(inputs_ids.shape[0], device=inputs_ids.device, dtype=torch.long)
            finish = torch.zeros(inputs_ids.shape[0], device=inputs_ids.device).bool()
            
            temperature = temperature[None].expand(inputs_ids.shape[0], -1)
            temperature = rearrange(temperature, "b n -> (b n) 1")
            print(temperature.shape, temperature)

            attention_mask_cache = torch.ones((inputs_ids.shape[0], inputs_ids.shape[1]+max_new_token,), dtype=torch.bool, device=inputs_ids.device)

            if attention_mask is not None:
                attention_mask_cache[:, :attention_mask.shape[1]] = attention_mask
            
            for i in tqdm(range(max_new_token)):
                model_input = self.prepare_inputs_for_generation(inputs_ids, 
                    outputs.past_key_values if i!=0 else None, 
                    attention_mask_cache[:, :inputs_ids.shape[1]], use_cache=True)
            
                if i == 0: 
                    model_input['inputs_embeds'] = self.emb_text(inputs_ids[:, :, 0])
                else:
                    model_input['inputs_embeds'] = self.emb_text(inputs_ids[:, :, 0][:,-1:]) # self.emb_text(inputs_ids[:, :, 0][-1:])
                
                model_input['input_ids'] = None
                # ['input_ids' None, 'position_ids' torch.Size([1, 1]), 'past_key_values' 20 2 torch.Size([1, 1, 12, 64], 'use_cache', 'attention_mask' torch.Size([1, 59]), 'inputs_embeds'] # torch.Size([1, 1, 768]))
                
                outputs = self.gpt.forward(**model_input, output_attentions=False) # return_attn = false odict_keys(['last_hidden_state', 'past_key_values'])
                
                hidden_states = outputs[0] # 🐻 [1, 57, 768]

                logits = self.head_text(hidden_states)
        
                logits = logits[:, -1].float()

                logits_token = inputs_ids[:, start_idx:, 0]
                    
                logits = logits / temperature
                
                for logitsProcessors in LogitsProcessors:
                    logits = logitsProcessors(logits_token, logits)
                    
                for logitsWarpers in LogitsWarpers:
                    logits = logitsWarpers(logits_token, logits)
                    
                if i < min_new_token:
                    logits[:, eos_token] = -torch.inf
                
                scores = F.softmax(logits, dim=-1)
            
                idx_next = torch.multinomial(scores, num_samples=1)
                
                finish = finish | (idx_next == eos_token).any(1)
                inputs_ids = torch.cat([inputs_ids, idx_next.unsqueeze(-1).expand(-1, -1, self.num_vq)], 1)

                end_idx = end_idx + (~finish).int()
            
                if finish.all():
                    break
            
            inputs_ids = [inputs_ids[idx, start_idx: start_idx+i] for idx, i in enumerate(end_idx.int())]
            inputs_ids = [i[:, 0] for i in inputs_ids]
            
            if not finish.all():
                self.logger.warn(f'Incomplete result. hit max_new_token: {max_new_token}')    
            
            return inputs_ids