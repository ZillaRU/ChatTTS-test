import ChatTTS
import torch
import torchaudio

chat = ChatTTS.Chat("tpu")
chat.load_models(local_path='./model_files')
wavs = chat.infer(["如果上天能再给我一次机会的话、我会对这个女孩说“我爱你”。如果非要在这份爱加上一个期限。我希望是,一万年!"], skip_refine_text=False, use_decoder=True)

wavs = chat.infer(["如果上天能再给我一次机会的话、我会对这个女孩说“我爱你”。如果非要在这份爱加上一个期限。我希望是,一万年!"], 
                  skip_refine_text=False, use_decoder=True, 
                  params_infer_code={'prompt':'[speed_7]',
                                     'spk_emb': chat.sample_random_speaker(9)})

wavs = chat.infer(["如果上天能再给我一次机会的话、我会对这个女孩说“我爱你”。如果非要在这份爱加上一个期限。我希望是,一万年!"], 
                  skip_refine_text=True, use_decoder=True, 
                  params_infer_code={'prompt':'[speed_7]',
                                     'spk_emb': chat.sample_random_speaker(7)})