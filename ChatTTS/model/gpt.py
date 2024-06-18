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
        # self.model.temperature = args.temperature
        # self.model.top_p = args.top_p
        # self.model.repeat_penalty = args.repeat_penalty
        # self.model.repeat_last_n = args.repeat_last_n
        # self.model.max_new_tokens = args.max_new_tokens
        # self.model.generation_mode = args.generation_mode
        # self.model.prompt_mode = args.prompt_mode
    # def prepare_inputs_for_generation(
    #     self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    # ):
    #     if past_key_values:
    #         input_ids = input_ids[:, -1:]
    #     position_ids = kwargs.get("position_ids", None)
    #     if attention_mask is not None and position_ids is None:
    #         # create position_ids on the fly for batch generation
    #         position_ids = attention_mask.long().cumsum(-1) - 1
    #         position_ids.masked_fill_(attention_mask == 0, 1)
    #         if past_key_values:
    #             position_ids = position_ids[:, -1].unsqueeze(-1)

    #     # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
    #     if inputs_embeds is not None and past_key_values is None:
    #         model_inputs = {"inputs_embeds": inputs_embeds}
    #     else:
    #         model_inputs = {"input_ids": input_ids}

    #     model_inputs.update(
    #         {
    #             "position_ids": position_ids,
    #             "past_key_values": past_key_values,
    #             "use_cache": kwargs.get("use_cache"),
    #             "attention_mask": attention_mask,
    #         }
    #     )
    #     return model_inputs
      
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
        # spk_idx就是inputs_ids中值为21143的下标
        spk_idx = torch.where(inputs_ids == 21143)[1]
        # spk_emb转成fp16，cpp按照原样接收内存值（格式是uint16）
        res = self.gpt.generate_code(inputs_ids, spk_idx, spk_emb, eos_token, temperature)
        breakpoint()
        print(res['inputs_ids'])
        # cpp中实际是从device mem拷贝的fp16数值（但vector定义中写的是uint16），这里直接按fp16读取
        res['hiddens'] = torch.from_numpy(np.frombuffer(res['hiddens'], dtype=np.float16)).to(dtype=torch.float32)
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
        inputs_ids = self.gpt.generate_text(inputs_ids_list, int(eos_token.item()), float(temperature.item()))
        return inputs_ids