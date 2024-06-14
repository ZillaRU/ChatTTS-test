import os
import torch
from tqdm import tqdm
torch._dynamo.config.cache_size_limit = 64
torch.set_float32_matmul_precision('high')
import torch.jit as jit
import torch.onnx as onnx
import ChatTTS


chat = ChatTTS.Chat()
chat.load_models()


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

NUM_OF_LAYERS = config.num_hidden_layers
HIDDEN_SIZE = config.hidden_size
NUM_ATTENTION_HEADS = config.num_attention_heads
NUM_KEY_VALUE_HEADS = config.num_key_value_heads
HEAD_DIM = HIDDEN_SIZE // NUM_ATTENTION_HEADS # 64
VOCAB_SIZE = config.vocab_size
SEQ_LENGTH = 512
folder = f"./tmp/onnx"


for param in chat.pretrain_models['gpt'].emb_text.parameters():
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
        return chat.pretrain_models['gpt'].emb_code(input_ids)

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


class Block(torch.nn.Module):
    def __init__(self, layer_id):
        super().__init__()
        self.layer_id = layer_id
        self.layer = layers[layer_id] # LlamaDecoderLayer
    def forward(self, hidden_states, position_ids, attention_mask):
        hidden_states, past_kv = self.layer(hidden_states=hidden_states,
                                            attention_mask=attention_mask,
                                            position_ids=position_ids,
                                            use_cache=True)
        present_k, present_v = past_kv
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

    def forward(self, hidden_states, position_ids, attention_mask, past_k,
                past_v):
        hidden_states, past_kv = self.layer(hidden_states,
                                            attention_mask,
                                            position_ids=position_ids,
                                            past_key_value=(past_k, past_v),
                                            use_cache=True)
        present_k, present_v = past_kv
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

# class LmHead_infer_text(torch.nn.Module):

#     def __init__(self):
#         super().__init__()

#     def forward(self, hidden_states):
#         hidden_states = gpt_model.norm(hidden_states)
#         m_logits = chat.pretrain_models['gpt'].head_text(hidden_states)
#         return m_logits


# class LmHead_infer_code(torch.nn.Module):

#     def __init__(self):
#         super().__init__()

#     def forward(self, hidden_states):
#         hidden_states = gpt_model.norm(hidden_states)
#         m_logits = torch.stack([chat.pretrain_models['gpt'].head_code[i](hidden_states) for i in range(chat.pretrain_models['gpt'].num_vq)], 3)
#         return m_logits

# def convert_lm_head_text():
#     model = LmHead_infer_text()
#     input = torch.randn(1, HIDDEN_SIZE)

#     torch.onnx.export(model, (input),
#                       f'{folder}/lm_head_text.onnx',
#                       verbose=False,
#                       input_names=['hidden_states'],
#                       output_names=['m_logits'],
#                       do_constant_folding=True,
#                       opset_version=15)
    
# def convert_lm_head_code():
#     model = LmHead_infer_code()
#     input = torch.randn(1, HIDDEN_SIZE)

#     torch.onnx.export(model, (input),
#                       f'{folder}/lm_head_code.onnx',
#                       verbose=False,
#                       input_names=['hidden_states'],
#                       output_names=['m_logits'],
#                       do_constant_folding=True,
#                       opset_version=15)

# create folder to store onnx
if not os.path.exists(folder):
    os.makedirs(folder)

# export models
print(f'Convert block & block_cache')
for i in tqdm(range(NUM_OF_LAYERS)):
    convert_block_cache(i)
    convert_block(i)

print(f'Convert embedding')
convert_embedding()

# print(f'Convert lm_head')
# convert_lm_head_code()
# convert_lm_head_text()
print("Done")