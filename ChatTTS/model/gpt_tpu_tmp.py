import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import logging
from tqdm import tqdm
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm
import sys
module_path = "./ChatTTS/model"
if module_path not in sys.path:
    sys.path.append(module_path)

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
        self.gpt.init([0], './chattts-llama_bf16_1dev_512.bmodel')
        self.num_vq = num_vq
        self.gpt.max_new_tokens = 512
        self.gpt.SEQLEN = 512
        self.gpt.top_p = 0.7
        # self.gpt.top_k = 20
        self.gpt.DEBUGGING = True
        self.gpt.temperature = 0.7
        self.gpt.repeat_penalty = 1.0

    def generate_code(
        self,
        inputs_ids,
        spk_emb,
        temperature, 
        eos_token, 
        attention_mask = None,
        max_new_token = 500, 
        min_new_token = 0,
        LogitsWarpers = [],
        LogitsProcessors = [],
        return_hidden=True,
    ):
        # spk_idx就是inputs_ids中值为21143的下标; 若不存在则为-1
        temp = torch.where(inputs_ids[0] == 21143)
        if temp[0].shape[0] == 0:
            spk_idx = -1
            spk_emb = list(range(768)) # NOT USED
        else:
            spk_idx = temp[0].item()
            # spk_emb转成fp16，cpp按照原样接收内存值（格式是uint16）
            spk_emb = list(spk_emb.to(dtype=torch.float16))

        with torch.no_grad():  
 
            hiddens = []
            
            start_idx, end_idx = inputs_ids.shape[1], torch.zeros(inputs_ids.shape[0], device=inputs_ids.device, dtype=torch.long)
            finish = torch.zeros(inputs_ids.shape[0], device=inputs_ids.device).bool()
            
            temperature = temperature[None].expand(inputs_ids.shape[0], -1)
            temperature = rearrange(temperature, "b n -> (b n) 1")

            curr_input_id = None
            for i in tqdm(range(max_new_token)):
                if i == 0:
                    logits, hidden = self.gpt.forward_first_code_core(inputs_ids[0].tolist(), spk_idx, spk_emb)
                    inputs_ids = inputs_ids.unsqueeze(2).expand(-1, -1, 4)
                    # breakpoint()
                else:
                    logits, hidden = self.gpt.forward_next_code_core(curr_input_id)
                    # breakpoint()
                
                hiddens.append(torch.tensor(hidden, dtype=torch.float32))
                logits = torch.tensor(logits).reshape(626, 4).transpose(0, 1)
                logits_token = rearrange(inputs_ids[:, start_idx:], "b c n -> (b n) c") # [1, 1, 4] [4,1]

                logits = logits / temperature
                
                for logitsProcessors in LogitsProcessors:
                    logits = logitsProcessors(logits_token, logits)
                    
                for logitsWarpers in LogitsWarpers:
                    logits = logitsWarpers(logits_token, logits)
                    
                if i < min_new_token:
                    logits[:, eos_token] = -torch.inf
                
                scores = F.softmax(logits, dim=-1)
            
                idx_next = torch.multinomial(scores, num_samples=1) # 每一行代表一个分布
                
                idx_next = rearrange(idx_next, "(b n) 1 -> b n", n=self.num_vq)
                finish = finish | (idx_next == eos_token).any(1)
                inputs_ids = torch.cat([inputs_ids, idx_next.unsqueeze(1)], 1)
                curr_input_id = inputs_ids[0, -1].int().tolist()
                end_idx = end_idx + (~finish).int()
                print(curr_input_id)
                if finish.all():
                    break
            
            inputs_ids = [inputs_ids[idx, start_idx: start_idx+i] for idx, i in enumerate(end_idx.int())]
            
            hiddens = torch.stack(hiddens, 1)
            hiddens = [hiddens[idx, :i] for idx, i in enumerate(end_idx.int())]
                    
            if not finish.all():
                self.logger.warn(f'Incomplete result. hit max_new_token: {max_new_token}')    
            breakpoint()
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
                print(idx_next.shape, idx_next)
                finish = finish | (idx_next == eos_token).any(1)
                inputs_ids = torch.cat([inputs_ids, idx_next.unsqueeze(-1).expand(-1, -1, self.num_vq)], 1)
                # breakpoint()

                end_idx = end_idx + (~finish).int()
            
                if finish.all():
                    break
            
            inputs_ids = [inputs_ids[idx, start_idx: start_idx+i] for idx, i in enumerate(end_idx.int())]
            inputs_ids = [i[:, 0] for i in inputs_ids]
            breakpoint()
            if not finish.all():
                self.logger.warn(f'Incomplete result. hit max_new_token: {max_new_token}')    
            
            return inputs_ids