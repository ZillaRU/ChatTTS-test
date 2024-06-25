import os
import torch
from tqdm import tqdm
torch._dynamo.config.cache_size_limit = 64
torch.set_float32_matmul_precision('high')
import torch.jit as jit
import torch.onnx as onnx
import ChatTTS
from transformers.generation import TopKLogitsWarper, TopPLogitsWarper
from ChatTTS.utils.infer_utils import CustomRepetitionPenaltyLogitsProcessorRepeat
import torch.nn.functional as F

chat = ChatTTS.Chat()
chat.load_models('local', local_path='./model_files')


gpt_model = chat.pretrain_models['gpt'].gpt.eval()

for param in gpt_model.parameters():
    param.requires_grad = False

'''
self.gpt.config
LlamaConfig {
  "attention_bias": false,
  "attention_dropout": 0.0,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "hidden_act": "silu",
  "hidden_size": 768,
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "max_position_embeddings": 4096,
  "mlp_bias": false,
  "model_type": "llama",
  "num_attention_heads": 12,
  "num_audio_tokens": 626,
  "num_hidden_layers": 20,
  "num_key_value_heads": 12,
  "num_text_tokens": null,
  "num_vq": 4,
  "pretraining_tp": 1,
  "rms_norm_eps": 1e-06,
  "rope_scaling": null,
  "rope_theta": 10000.0,
  "spk_KL": false,
  "spk_emb_dim": 192,
  "tie_word_embeddings": false,
  "transformers_version": "4.41.2",
  "use_cache": false,
  "vocab_size": 32000
}
'''

config = gpt_model.config
layers = gpt_model.layers
model_norm = gpt_model.norm

NUM_OF_LAYERS = config.num_hidden_layers
HIDDEN_SIZE = config.hidden_size
NUM_ATTENTION_HEADS = config.num_attention_heads
NUM_KEY_VALUE_HEADS = config.num_key_value_heads
HEAD_DIM = HIDDEN_SIZE // NUM_ATTENTION_HEADS # 64
TEXT_VOCAB_SIZE = 21178
AUDIO_VOCAB_SIZE = 626 # config.vocab_size
SEQ_LENGTH = 512
folder = f"./tmp/onnx"


for param in chat.pretrain_models['gpt'].emb_text.parameters():
    param.requires_grad = False

for param in chat.pretrain_models['gpt'].emb_code.parameters():
    param.requires_grad = False

for param in chat.pretrain_models['gpt'].head_code.parameters():
    param.requires_grad = False

for param in chat.pretrain_models['gpt'].head_text.parameters():
    param.requires_grad = False

class EmbeddingText(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    def forward(self, input_ids):
        return chat.pretrain_models['gpt'].emb_text(input_ids)

def convert_embedding_text():
    model = EmbeddingText()
    input_ids = torch.tensor([range(SEQ_LENGTH)])

    torch.onnx.export(model, (input_ids),
                      f'{folder}/embedding_text.onnx',
                      verbose=False,
                      input_names=['input_ids'],
                      output_names=['input_embed'],
                      do_constant_folding=True,
                      opset_version=15)

class EmbeddingCode(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    def forward(self, input_ids):
        input_ids = input_ids.unsqueeze(2).expand(-1, -1, chat.pretrain_models['gpt'].num_vq) # for forward_first_code
        code_emb = [chat.pretrain_models['gpt'].emb_code[i](input_ids[:,:,i]) for i in range(chat.pretrain_models['gpt'].num_vq)]
        return torch.stack(code_emb, 2).sum(2)

def convert_embedding_code():
    model = EmbeddingCode()
    input_ids = torch.tensor([range(SEQ_LENGTH)])

    torch.onnx.export(model, (input_ids),
                      f'{folder}/embedding_code.onnx',
                      verbose=False,
                      input_names=['input_ids'],
                      output_names=['input_embed'],
                      do_constant_folding=True,
                      opset_version=15)

class EmbeddingCodeCache(torch.nn.Module):  # for forward_next_code
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    def forward(self, input_ids):
        code_emb = [chat.pretrain_models['gpt'].emb_code[i](input_ids[:,:,i]) for i in range(chat.pretrain_models['gpt'].num_vq)]
        return torch.stack(code_emb, 2).sum(2)

def convert_embedding_code_cache():
    model = EmbeddingCodeCache()
    input_ids = torch.tensor([[range(chat.pretrain_models['gpt'].num_vq)]])
    print(input_ids.shape)

    torch.onnx.export(model, (input_ids),
                      f'{folder}/embedding_code_cache.onnx',
                      verbose=False,
                      input_names=['input_ids'],
                      output_names=['input_embed'],
                      do_constant_folding=True,
                      opset_version=15)

class Block(torch.nn.Module):
    def __init__(self, layer_id):
        super().__init__()
        self.layer_id = layer_id
        self.layer = layers[layer_id] # LlamaDecoderLayer
        self.norm = model_norm
    def forward(self, hidden_states, position_ids, attention_mask):
        hidden_states, past_kv = self.layer(hidden_states=hidden_states,
                                            attention_mask=attention_mask,
                                            position_ids=position_ids,
                                            use_cache=True)
        present_k, present_v = past_kv
        if(self.layer_id == NUM_OF_LAYERS - 1):
            hidden_states = self.norm(hidden_states)
        return hidden_states, present_k, present_v

def convert_block(layer_id):
    model = Block(layer_id)
    hidden_states = torch.randn((1, SEQ_LENGTH, HIDDEN_SIZE))
    position_ids = torch.tensor([range(SEQ_LENGTH)], dtype=torch.long)
    attention_mask = -1000 * torch.ones((1, 1, SEQ_LENGTH, SEQ_LENGTH), dtype=torch.float32).triu(diagonal=1)
    model(hidden_states, position_ids, attention_mask)
    torch.onnx.export(
        model, (hidden_states, position_ids, attention_mask),
        f'{folder}/block_{layer_id}.onnx',
        verbose=False,
        input_names=['input_states', 'position_ids', 'attention_mask'],
        output_names=['hidden_states', 'past_k', 'past_v'],
        do_constant_folding=True,
        opset_version=15)

class BlockCache(torch.nn.Module):

    def __init__(self, layer_id):
        super().__init__()
        self.layer_id = layer_id
        self.layer = layers[layer_id]
        self.norm = model_norm

    def forward(self, hidden_states, position_ids, attention_mask, past_k,
                past_v):
        hidden_states, past_kv = self.layer(hidden_states,
                                            attention_mask,
                                            position_ids=position_ids,
                                            past_key_value=(past_k, past_v),
                                            use_cache=True)
        present_k, present_v = past_kv
        if(self.layer_id == NUM_OF_LAYERS - 1):
            hidden_states = self.norm(hidden_states)
        return hidden_states, present_k, present_v

def convert_block_cache(layer_id):
    model = BlockCache(layer_id)
    hidden_states = torch.randn((1, 1, HIDDEN_SIZE))
    position_ids = torch.tensor([range(1)], dtype=torch.long) ############## shape???
    attention_mask = -1000 * torch.ones((1, 1, 1, SEQ_LENGTH+1), dtype=torch.float32).triu(diagonal=1)
    past_k = torch.randn((1, SEQ_LENGTH, NUM_ATTENTION_HEADS, HEAD_DIM))
    past_v = torch.randn((1, SEQ_LENGTH, NUM_ATTENTION_HEADS, HEAD_DIM))

    torch.onnx.export(
        model, (hidden_states, position_ids, attention_mask, past_k, past_v),
        f'{folder}/block_cache_{layer_id}.onnx',
        verbose=False,
        input_names=[
            'input_states', 'position_ids', 'attention_mask', 'history_k',
            'history_v'
        ],
        output_names=['hidden_states', 'past_k', 'past_v'],
        do_constant_folding=True,
        opset_version=15)

class GreedyHead(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, m_logits):
        _, token = torch.topk(m_logits.float(), 1)
        return token

def convert_greedy_head_text():   
    model = GreedyHead()
    m_logits = torch.randn(1, TEXT_VOCAB_SIZE)

    torch.onnx.export(
        model, (m_logits),
        f'{folder}/greedy_head_text.onnx',
        verbose=False,
        input_names=['m_logits'],
        output_names=['token'],
        do_constant_folding=True,
        opset_version=15)
    
def convert_greedy_head_code():   
    model = GreedyHead()
    m_logits = torch.randn(1, AUDIO_VOCAB_SIZE, chat.pretrain_models['gpt'].num_vq)

    torch.onnx.export(
        model, (m_logits),
        f'{folder}/greedy_head_code.onnx',
        verbose=False,
        input_names=['m_logits'],
        output_names=['token'],
        do_constant_folding=True,
        opset_version=15)

# refs:https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py
class PenaltySampleHeadText(torch.nn.Module):
    def __init__(self, top_k = 50, min_tokens_to_keep = 5):
        super().__init__()
        self.top_k = top_k
        self.min_tokens_to_keep = min_tokens_to_keep
        self.keep_matrix = torch.zeros((1, self.top_k), dtype=torch.bool)
        self.keep_matrix[0, :self.min_tokens_to_keep] = True

    def forward(self, m_logits, input_ids, top_p, temperature, penalty):
        # repeat penalty
        logits = torch.gather(m_logits, 1, input_ids)
        logits = torch.where(logits < 0, logits * penalty, logits / penalty)
        m_logits.scatter_(1, input_ids, logits)

        # top_k
        logits, token = torch.topk(m_logits.float(), self.top_k, dim=1)

        # temperature
        logits = logits / temperature

        # top_p
        cumulative_probs = logits.softmax(dim=1).cumsum(dim=1)
        mask = cumulative_probs < top_p
        mask = mask + self.keep_matrix
        filtered_logits = torch.where(mask, logits, torch.FloatTensor([-1000.]))
        probs = filtered_logits.softmax(dim=1)
        return probs, token
    
def convert_penalty_sample_head_text(VOCAB_SIZE):   
    model = PenaltySampleHeadText(top_k=20, min_tokens_to_keep=3)
    m_logits = torch.randn(1, VOCAB_SIZE) ### for text generation: VOCAB_SIZE
    input_ids = torch.tensor([range(SEQ_LENGTH)])
    top_p = torch.tensor([0.7])
    temperature = torch.tensor([0.7])
    penalty = torch.tensor([0.98])

    torch.onnx.export(
        model, (m_logits, input_ids, top_p, temperature, penalty),
        f'{folder}/penalty_sample_head_text.onnx',
        verbose=False,
        input_names=[
            'm_logits', 'input_ids', 'top_p', 'temperature',
            'penalty'
        ],
        output_names=['probs', 'token'],
        do_constant_folding=True,
        opset_version=15)

class TTSSampleHeadCode(torch.nn.Module):
    def __init__(self, vacab_size=AUDIO_VOCAB_SIZE, window_size=16, repetition_penalty=None, top_P=None, top_K=None):
        super().__init__()
        self.LogitsWarpers = []
        if top_P is not None:
            self.LogitsWarpers.append(TopPLogitsWarper(top_P, min_tokens_to_keep=3, filter_value=-1000.0))
        if top_K is not None:
            self.LogitsWarpers.append(TopKLogitsWarper(top_K, min_tokens_to_keep=3, filter_value=-1000.0))
            
        self.LogitsProcessors = []
        if repetition_penalty is not None and repetition_penalty != 1:
            self.LogitsProcessors.append(CustomRepetitionPenaltyLogitsProcessorRepeat(repetition_penalty, vacab_size-1, window_size))
        

    def forward(self, m_logits, input_ids, valid_len, penalty, temperature = 1.0):
        # m_logits [626, 4] -> [4, 626]  # input_ids [4, seq_len_from_start]
        m_logits = m_logits.transpose(0, 1)
        m_logits = m_logits / temperature
        
        for logitsProcessor in self.LogitsProcessors:
            m_logits = logitsProcessor(input_ids, m_logits, valid_seq_len=valid_len)
            
        for logitsWarper in self.LogitsWarpers:
            logitsWarper.penalty = penalty
            m_logits = logitsWarper(input_ids, m_logits)
            
        # if i < min_new_token: # 0
        #     logits[:, eos_token] = -torch.inf
        
        m_logits = F.softmax(m_logits, dim=1)
        return m_logits # [4, 626]

def convert_chattts_sample_head_code(WINDOW_SIZE=16):
    model = TTSSampleHeadCode(top_K=20, top_P=0.7, vacab_size=AUDIO_VOCAB_SIZE, repetition_penalty=1.05) # 1.05 for code, 1.0 for text
    m_logits = torch.randn(AUDIO_VOCAB_SIZE, chat.pretrain_models['gpt'].num_vq) # 626,4
    input_ids = torch.tensor([range(WINDOW_SIZE)]).expand(chat.pretrain_models['gpt'].num_vq, -1)
    valid_seq_len = torch.tensor([10]).int()
    temperature = torch.tensor([0.7])
    penalty = torch.tensor([1.05]) 
    import numpy as np
    np.savez(f'{folder}/chattts_sample_head_code_input.npz', m_logits=m_logits, input_ids=input_ids, valid_seq_len=valid_seq_len, penalty=penalty, temperature=temperature)
    torch.onnx.export(
        model, (m_logits, input_ids, valid_seq_len, penalty, temperature),
        f'{folder}/chattts_sample_head_code.onnx',
        verbose=False,
        input_names=[
            'm_logits', 'input_ids', 'valid_seq_len', 'penalty', 'temperature'
        ],
        output_names=['probs'],
        do_constant_folding=True,
        opset_version=15)

class PenaltySampleHeadCode(torch.nn.Module):
    def __init__(self, top_k = 50, min_tokens_to_keep = 5):
        super().__init__()
        self.top_k = top_k
        self.min_tokens_to_keep = min_tokens_to_keep
        self.keep_matrix = torch.zeros((1, self.top_k, chat.pretrain_models['gpt'].num_vq), dtype=torch.bool)
        self.keep_matrix[0, :self.min_tokens_to_keep, :] = True

    def forward(self, m_logits, input_ids, top_p, temperature, penalty):
        # m_logits: [1, VOCAB_SIZE, NUM_VQ]
        logits = torch.gather(m_logits, 1, input_ids)
        logits = torch.where(logits < 0, logits * penalty, logits / penalty)
        m_logits.scatter_(1, input_ids, logits)

        # top_k
        logits, token = torch.topk(m_logits.float(), self.top_k, dim=1)

        # temperature
        logits = logits / temperature

        # top_p
        cumulative_probs = logits.softmax(dim=1).cumsum(dim=1)
        mask = cumulative_probs < top_p
        mask = mask + self.keep_matrix
        filtered_logits = torch.where(mask, logits, torch.FloatTensor([-1000.]))
        probs = filtered_logits.softmax(dim=1)
        return torch.transpose(probs, -2, -1), torch.transpose(token, -2, -1) ## [xxx, 4] --> [4, xxx]

def convert_penalty_sample_head_code(VOCAB_SIZE):   
    model = PenaltySampleHeadCode(top_k=20, min_tokens_to_keep=3)
    m_logits = torch.randn(1, VOCAB_SIZE, chat.pretrain_models['gpt'].num_vq)
    input_ids = torch.tensor([range(SEQ_LENGTH)]).unsqueeze(-1).expand(-1, -1, chat.pretrain_models['gpt'].num_vq)
    top_p = torch.tensor([0.7])
    temperature = torch.tensor([0.7])
    penalty = torch.tensor([0.98])

    torch.onnx.export(
        model, (m_logits, input_ids, top_p, temperature, penalty),
        f'{folder}/penalty_sample_head_code.onnx',
        verbose=False,
        input_names=[
            'm_logits', 'input_ids', 'top_p', 'temperature',
            'penalty'
        ],
        output_names=['probs', 'token'],
        do_constant_folding=True,
        opset_version=15)

class LmHead_infer_text(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states):
        # hidden_states = gpt_model.norm(hidden_states)
        m_logits = chat.pretrain_models['gpt'].head_text(hidden_states)
        return m_logits


class LmHead_infer_code(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, hidden_states):
        # hidden_states = gpt_model.norm(hidden_states)
        breakpoint()
        m_logits = torch.stack([chat.pretrain_models['gpt'].head_code[i](hidden_states) for i in range(chat.pretrain_models['gpt'].num_vq)], 2)
        return m_logits

def convert_lm_head_text():
    model = LmHead_infer_text()
    input = torch.randn(1, HIDDEN_SIZE)

    torch.onnx.export(model, (input),
                      f'{folder}/lm_head_text.onnx',
                      verbose=False,
                      input_names=['hidden_states'],
                      output_names=['m_logits'],
                      do_constant_folding=True,
                      opset_version=15)
    
def convert_lm_head_code():
    model = LmHead_infer_code()
    input = torch.randn(1, HIDDEN_SIZE)
    print(input.shape)
    torch.onnx.export(model, (input),
                      f'{folder}/lm_head_code.onnx',
                      verbose=False,
                      input_names=['hidden_states'],
                      output_names=['m_logits'],
                      do_constant_folding=True,
                      opset_version=15)

# create folder to store onnx
if not os.path.exists(folder):
    os.makedirs(folder)


# export models
# print(f'Convert block & block_cache')
# for i in tqdm(range(NUM_OF_LAYERS)):
#     convert_block_cache(i)
#     convert_block(i)

# print(f'Convert embedding')
# convert_embedding_text()
# convert_embedding_code()
# convert_embedding_code_cache()

# print(f'Convert lm_head')
# convert_lm_head_code()
# convert_lm_head_text()

# print(f'Convert greedy_head')
# convert_greedy_head_text()
# convert_greedy_head_code()

print(f'Convert penalty_sample_head')
# convert_penalty_sample_head_text(TEXT_VOCAB_SIZE)
# convert_penalty_sample_head_code(AUDIO_VOCAB_SIZE)
convert_chattts_sample_head_code()
print("Done")