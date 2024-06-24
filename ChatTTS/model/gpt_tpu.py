import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
module_path = "./ChatTTS/model"
if module_path not in sys.path:
    sys.path.append(module_path)

import numpy as np
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
        self.gpt.init([0], './chattts-llama_int8_1dev_512.bmodel')
        self.num_vq = num_vq
        self.gpt.max_new_tokens = 512
        self.gpt.SEQLEN = 512
        self.gpt.top_p = 0.7
        # self.gpt.top_k = 20
        self.gpt.DEBUGGING = True
        self.gpt.temperature = 0.7
        self.gpt.repeat_penalty = 1.0
        self.gpt.repeat_last_n = 3

    def generate_code(
        self,
        inputs_ids,
        spk_emb,
        temperature, 
        eos_token, 
        attention_mask = None,
        max_new_token = 512, 
        min_new_token = 0,
        LogitsWarpers = [],
        LogitsProcessors = [],
        return_hidden=True,
    ):
        self.gpt.temperature = temperature
        self.gpt.repeat_penalty = 1.05
        # spk_idx就是inputs_ids中值为21143的下标; 若不存在则为-1
        breakpoint()
        temp = torch.where(inputs_ids[0] == 21143)
        if temp[0].shape[0] == 0:
            spk_idx = -1
            spk_emb = list(range(768)) # NOT USED
        else:
            spk_idx = temp[0].item()
            # spk_emb转成fp16，cpp按照原样接收内存值（格式是uint16）
            spk_emb = list(spk_emb.to(dtype=torch.float16))

        inputs_ids_list = inputs_ids[0].tolist()
        
        res = self.gpt.generate_code(inputs_ids_list, spk_idx, spk_emb, eos_token, temperature)
        print(res['tokens'])
        res['ids'] = torch.tensor(res['tokens'], dtype=torch.int64).unsqueeze(0)
        print(res['ids'].shape)
        hiddens_np = np.array(res['hiddens'], dtype=np.uint16).view(np.float16).astype(np.float32)/100
        breakpoint()
        # cpp中实际是从device mem拷贝的fp16数值（但vector定义中写的是uint16），这里直接按fp16读取
        res['hiddens'] = torch.from_numpy(hiddens_np).unsqueeze(0)
        return res
    
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
        breakpoint()
        inputs_ids_list = inputs_ids[0].tolist()
        self.gpt.temperature = temperature
        self.gpt.repeat_penalty = 1.0
        inputs_ids = self.gpt.generate_text(inputs_ids_list, eos_token, temperature)
        # inputs_ids_list += inputs_ids
        # inputs_ids = torch.tensor(inputs_ids_list, dtype=torch.int64).unsqueeze(0).unsqueeze(0)
        inputs_ids = torch.tensor(inputs_ids, dtype=torch.int64).unsqueeze(0).unsqueeze(0)
        print(inputs_ids.shape)
        breakpoint()
        return inputs_ids