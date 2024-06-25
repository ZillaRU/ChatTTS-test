from transformers.generation import TopKLogitsWarper, TopPLogitsWarper
from ChatTTS.utils.infer_utils import CustomRepetitionPenaltyLogitsProcessorRepeat
import torch.nn.functional as F
import torch

folder='debug'

class TTSSampleHeadCode(torch.nn.Module):
    def __init__(self, vacab_size=626, window_size=16, repetition_penalty=None, top_P=None, top_K=None):
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
    model = TTSSampleHeadCode(top_K=20, top_P=0.7, vacab_size=626, repetition_penalty=1) # 1.05 for code, 1.0 for text
    m_logits = torch.randn(626, 4) # 626,4
    input_ids = torch.tensor([range(WINDOW_SIZE)]).expand(4, -1)
    valid_seq_len = torch.tensor([10]).int()
    temperature = torch.tensor([0.7])
    penalty = torch.tensor([1.05]) 
    import numpy as np
    np.savez(f'{folder}/chattts_sample_head_code_input1.npz', m_logits=m_logits, input_ids=input_ids, valid_seq_len=valid_seq_len, penalty=penalty, temperature=temperature)
    torch.onnx.export(
        model, (m_logits, input_ids, valid_seq_len, penalty, temperature),
        f'{folder}/chattts_sample_head_code1.onnx',
        verbose=False,
        input_names=[
            'm_logits', 'input_ids', 'valid_seq_len', 'penalty', 'temperature'
        ],
        output_names=['probs'],
        do_constant_folding=True,
        opset_version=15)


class TTSSampleHeadCode_Part(torch.nn.Module):
    def __init__(self, vacab_size=626, window_size=16, repetition_penalty=None, top_P=None, top_K=None):
        super().__init__()
        self.LogitsProcessors = CustomRepetitionPenaltyLogitsProcessorRepeat(repetition_penalty, vacab_size-1, window_size)
        

    def forward(self, m_logits, input_ids, valid_len):
        # m_logits [626, 4] -> [4, 626]  # input_ids [4, seq_len_from_start]
        m_logits = m_logits.transpose(0, 1)
        m_logits = self.LogitsProcessors(input_ids, m_logits, valid_seq_len=valid_len)
        return m_logits # [4, 626]

def convert_chattts_sample_head_code_part(WINDOW_SIZE=16):
    model = TTSSampleHeadCode_Part(top_K=20, top_P=0.7, vacab_size=626, repetition_penalty=1.05) # 1.05 for code, 1.0 for text
    m_logits = torch.randn(626, 4) # 626,4
    input_ids = torch.tensor([range(WINDOW_SIZE)]).expand(4, -1)
    valid_seq_len = torch.tensor([10]).int()
    import numpy as np
    np.savez(f'{folder}/chattts_sample_head_code_input2.npz', m_logits=m_logits, input_ids=input_ids, valid_seq_len=valid_seq_len)
    torch.onnx.export(
        model, (m_logits, input_ids, valid_seq_len),
        f'{folder}/chattts_sample_head_code2.onnx',
        verbose=False,
        input_names=[
            'm_logits', 'input_ids', 'valid_seq_len'
        ],
        output_names=['probs'],
        do_constant_folding=True,
        opset_version=15)

# convert_chattts_sample_head_code()
convert_chattts_sample_head_code_part()