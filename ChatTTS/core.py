
import os
import logging
from functools import partial
from omegaconf import OmegaConf
from npuengine import EngineOV
import torch
from .model.gpt_tpu_tmp import GPT_warpper
from .model.dvae import DVAE
from .model.vocos_spectral_ops import ISTFT
from .utils.infer_utils import count_invalid_characters, detect_language, apply_character_map, apply_half2full_map

logging.basicConfig(level = logging.INFO)


class Chat:
    def __init__(self, device = 'cpu'):
        self.device = device
        self.pretrain_models = {}
        self.normalizer = {}
        self.logger = logging.getLogger(__name__)
        self.postprocess = ISTFT(n_fft=1024, hop_length=256, win_length=1024, padding='center')

    def check_model(self, level = logging.INFO, use_decoder = False):
        not_finish = False
        check_list = ['vocos', 'gpt', 'tokenizer']
        
        if use_decoder:
            check_list.append('decoder')
        else:
            check_list.append('dvae')
            
        for module in check_list:
            if module not in self.pretrain_models:
                self.logger.log(logging.WARNING, f'{module} not initialized.')
                not_finish = True
                
        if not not_finish:
            self.logger.log(level, f'All initialized.')
            
        return not not_finish
        
    def load_models(self, local_path='<LOCAL_PATH>'):
        self._load(**{k: os.path.join(local_path, v) for k, v in OmegaConf.load(os.path.join(local_path, 'config', 'path.yaml')).items()})
        
    def _load(
        self, 
        vocos_ckpt_path: str = None,
        gpt_ckpt_path: str = None,
        decoder_ckpt_path: str = None,
        tokenizer_path: str = None,
        dvae_config_path: str = None,
        dvae_ckpt_path: str = None,
        tpu_id: int = 0
    ):

        device = torch.device('cpu')
            
        if vocos_ckpt_path:
            vocos = EngineOV(vocos_ckpt_path, device_id=tpu_id)
            self.pretrain_models['vocos'] = vocos
            self.logger.log(logging.INFO, 'vocos loaded.')
        
        if dvae_config_path:
            cfg = OmegaConf.load(dvae_config_path)
            dvae = DVAE(**cfg).to(device).eval()
            assert dvae_ckpt_path, 'dvae_ckpt_path should not be None'
            dvae.load_state_dict(torch.load(dvae_ckpt_path, map_location='cpu'))
            self.pretrain_models['dvae'] = dvae
            self.logger.log(logging.INFO, 'dvae loaded.')
            
        if gpt_ckpt_path:
            gpt = GPT_warpper(gpt_bmodel_path=gpt_ckpt_path)
            self.pretrain_models['gpt'] = gpt

            spk_stat_path = os.path.join(os.path.dirname(gpt_ckpt_path), 'spk_stat.pt')
            assert os.path.exists(spk_stat_path), f'Missing spk_stat.pt: {spk_stat_path}'
            self.pretrain_models['spk_stat'] = torch.load(spk_stat_path).to(device)
            self.logger.log(logging.INFO, 'gpt loaded.')
            
        if decoder_ckpt_path:
            decoder = EngineOV(decoder_ckpt_path, device_id=tpu_id)
            self.pretrain_models['decoder'] = decoder
            self.logger.log(logging.INFO, 'decoder loaded.')
        
        if tokenizer_path:
            tokenizer = torch.load(tokenizer_path, map_location='cpu')
            tokenizer.padding_side = 'left'
            self.pretrain_models['tokenizer'] = tokenizer
            self.logger.log(logging.INFO, 'tokenizer loaded.')
            
        self.check_model()
    
    def infer(
        self, 
        text, 
        skip_refine_text=False, 
        refine_text_only=False, 
        params_refine_text={}, 
        params_infer_code={'prompt':'[speed_5]'}, 
        use_decoder=True,
        do_text_normalization=True,
        lang=None,
    ):
        if self.device == 'cpu':
            from .infer.api import refine_text, infer_code
        else: 
            from .infer.api_tpu import refine_text, infer_code
        
        assert self.check_model(use_decoder=use_decoder)
        
        if not isinstance(text, list): 
            text = [text]
        
        if do_text_normalization:
            for i, t in enumerate(text):
                _lang = detect_language(t) if lang is None else lang
                self.init_normalizer(_lang)
                text[i] = self.normalizer[_lang](t)
                if _lang == 'zh':
                    text[i] = apply_half2full_map(text[i])
            
        for i, t in enumerate(text):
            invalid_characters = count_invalid_characters(t)
            if len(invalid_characters):
                self.logger.log(logging.WARNING, f'Invalid characters found! : {invalid_characters}')
                text[i] = apply_character_map(t)
                
        if not skip_refine_text:
            text_tokens = refine_text(self.pretrain_models, text, **params_refine_text) # ['ids']
            text_tokens = [i[i < self.pretrain_models['tokenizer'].convert_tokens_to_ids('[break_0]')] for i in text_tokens] #21147
            text = self.pretrain_models['tokenizer'].batch_decode(text_tokens)
            print('refine text', text)
            if refine_text_only:
                return text
            
        text = [params_infer_code.get('prompt', '') + i for i in text]
        params_infer_code.pop('prompt', '')
        result = infer_code(self.pretrain_models, text, **params_infer_code, return_hidden=use_decoder)
        
        if use_decoder:
            breakpoint()
            mel_spec = []
            _cut = []
            for i in result['hiddens']:
                i = i[None].permute(0,2,1)
                _cut.append(i.shape[-1])
                if(i.shape[-1] < 1024):
                    _pad = torch.zeros((i.shape[0], i.shape[1], 1024 - i.shape[-1]))
                    i = torch.cat([i, _pad], dim=2)
                elif i.shape[-1] > 1024:
                    i = i[:,:, :1024]
                    self.logger.warning('the dec mel_spec is larger than 1024')
                breakpoint()
                mel_spec.append(torch.from_numpy(self.pretrain_models['decoder']([i.numpy()])[0])) # out 1, 100, 2048
        else:
            mel_spec = [self.pretrain_models['dvae'](i[None].permute(0, 2, 1)) for i in result['ids']]
        print('decoder out permute / vocos input', [i.shape for i in mel_spec])
        wavs = []
        for idx, i in enumerate(mel_spec):
            # i padding from [1, 100, x] to [1,100,2048]
            ori_len = i.shape[-1]
            if i.shape[-1] < 2048:
                _pad = torch.zeros((i.shape[0], i.shape[1], 2048 - i.shape[-1]))
                i = torch.cat([i, _pad], dim=2).numpy()
            elif i.shape[-1] > 2048:
                i = i[:,:, :2048].numpy()
                self.logger.warning('the mel_spec is larger than 2048')
            else:
                i = i.numpy()
            mag, x, y = self.pretrain_models['vocos']([i])
            mag = torch.from_numpy(mag)
            x = torch.from_numpy(x)
            y = torch.from_numpy(y)
            S = mag * (x + 1j * y)
            audio = self.postprocess(S)
            if _cut[idx] < 1024: 
                audio = audio[:, :int(_cut[idx]/1024*audio.shape[1])]
            wavs.append(audio.numpy())
        breakpoint()
        return wavs  # [1,250624] #/256
    
    def sample_random_speaker(self, seed):
        torch.manual_seed(seed)
        dim = 768
        std, mean = self.pretrain_models['spk_stat'].chunk(2)
        return torch.randn(dim, device=std.device) * std + mean
    
    def init_normalizer(self, lang):
        
        if lang not in self.normalizer:
            if lang == 'zh':
                try:
                    from tn.chinese.normalizer import Normalizer
                except:
                    self.logger.log(logging.WARNING, f'Package WeTextProcessing not found! \
                        Run: conda install -c conda-forge pynini=2.1.5 && pip install WeTextProcessing')
                self.normalizer[lang] = Normalizer().normalize
            else:
                try:
                    from nemo_text_processing.text_normalization.normalize import Normalizer
                except:
                    self.logger.log(logging.WARNING, f'Package nemo_text_processing not found! \
                        Run: conda install -c conda-forge pynini=2.1.5 && pip install nemo_text_processing')
                self.normalizer[lang] = partial(Normalizer(input_case='cased', lang=lang).normalize, verbose=False, punct_post_process=True)

