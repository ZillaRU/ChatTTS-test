import ChatTTS
import torch
import torchaudio
import pdb; pdb.set_trace()

chat = ChatTTS.Chat()
chat.load_models(source='local', local_path='./model_files')
wavs = chat.infer(["I have to say, 无法理解你的意思。"]) #, skip_refine_text=True)
torchaudio.save("output1.wav", torch.from_numpy(wavs[0]), 24000)