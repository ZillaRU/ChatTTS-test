import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import logging
from tqdm import tqdm
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import numpy as np

module_path = "./ChatTTS/model"
if module_path not in sys.path:
    sys.path.append(module_path)

class GPT_warpper(nn.Module):
    def __init__(
        self, 
        gpt_bmodel_path,
        num_vq=4,
        devid=0
        ):
        super().__init__()
        import llama
        self.logger = logging.getLogger(__name__)
        self.gpt = llama.TTSLlama()
        self.gpt.init([devid], gpt_bmodel_path)
        self.num_vq = num_vq
        self.gpt.max_new_tokens = 512
        self.gpt.SEQLEN = 512
        self.gpt.top_p = 0.7
        self.gpt.repeat_last_n = 512
        # self.gpt.top_k = 20
        self.gpt.DEBUGGING = False
        self.gpt.temperature = 0.7
        self.gpt.repeat_penalty = 1.0

    def generate_code(
        self,
        inputs_ids,
        spk_emb,
        temperature, 
        eos_token, 
        attention_mask = None,
        max_new_token = 400, 
        min_new_token = 0,
        LogitsWarpers = [],
        LogitsProcessors = [],
        return_hidden=True,
    ):
        # self.gpt.DEBUGGING = True
        # self.gpt.temperature = temperature[0].item()
        # self.gpt.repeat_penalty = 1.05
        # # spk_idx就是inputs_ids中值为21143的下标; 若不存在则为-1
        # temp = torch.where(inputs_ids[0] == 21143)
        # if temp[0].shape[0] == 0:
        #     spk_idx = -1
        #     spk_emb = list(range(768))
        #     self.logger.info("Not set speaker")
        # else:
        #     spk_idx = temp[0].item()
        #     # spk_emb转成fp16，cpp按照原样接收内存值（格式是uint16）
        #     spk_emb = spk_emb[0].tolist()

        # inputs_ids_list = inputs_ids[0].tolist()
        # breakpoint()
        # res = self.gpt.generate_code(inputs_ids_list, spk_idx, spk_emb, eos_token, temperature[0].item())
        # print(res['tokens'])
        # res['ids'] = torch.tensor(res['tokens'], dtype=torch.int64).unsqueeze(0)
        # print(res['ids'].shape)
        # hiddens_np = np.array(res['hiddens'], dtype=np.float32)
        # # cpp中实际是从device mem拷贝的fp16数值（但vector定义中写的是uint16），这里直接按fp16读取
        # res['hiddens'] = torch.from_numpy(hiddens_np).unsqueeze(0)
        # return res
    
        # spk_idx就是inputs_ids中值为21143的下标; 若不存在则为-1
        temp = torch.where(inputs_ids[0] == 21143)
        if temp[0].shape[0] == 0:
            spk_idx = -1
            spk_emb = list(range(768))
            self.logger.info("Not set speaker")
        else:
            spk_idx = temp[0].item()
            # spk_emb转成fp16，cpp按照原样接收内存值（格式是uint16）
            spk_emb = spk_emb[0].tolist()

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
                else:
                    logits, hidden = self.gpt.forward_next_code_core(curr_input_id)
                
                hiddens.append(torch.tensor(hidden, dtype=torch.float32).unsqueeze(0))
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
                # print(curr_input_id)
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
        max_new_token = 500, 
        min_new_token = 0,
        LogitsWarpers = [],
        LogitsProcessors = [],
        return_hidden=False,
    ):
        inputs_ids_list = inputs_ids[0].tolist()
        if isinstance(eos_token, torch.Tensor):
            eos_token = eos_token.item()
        if isinstance(temperature, torch.Tensor):
            temperature = temperature.item()
        self.gpt.temperature = temperature
        self.gpt.repeat_penalty = 1.0
        inputs_ids = self.gpt.generate_text(inputs_ids_list, eos_token, temperature)
        inputs_ids = torch.tensor(inputs_ids, dtype=torch.int64).unsqueeze(0).unsqueeze(0)
        return inputs_ids