import ChatTTS
import torch
import torchaudio

chat = ChatTTS.Chat("tpu")
chat.load_models(source='local', local_path='./model_files')
wavs = chat.infer(["I have to say, 绝了。"], skip_refine_text=True, use_decoder=False)
torchaudio.save("output1.wav", torch.from_numpy(wavs[0]), 24000)