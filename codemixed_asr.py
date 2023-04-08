from datasets import load_dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
from jiwer import wer
import librosa
import os
import torchaudio
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from evaluate import load
import re



class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, audio_path, text_file):
        self.audio = []
        self.text = []
        for file in sorted(os.listdir(audio_path)):
            if file.endswith(".mp3"):
                self.audio_path = os.path.join(audio_path, file)
                audio, sampling_rat = librosa.load(self.audio_path,sr=16000)
                self.audio.append(audio)

        with open(text_file, "r") as f:
            text = f.readlines()
            for line in text:
                l = line.strip().split("_")
                l = "_".join(l[3:])
                l = re.sub(chars_to_ignore_regex, '', l).lower()
                self.text.append(l)
        
    def __len__(self):
        return len(self.text)
    
    def __getitem__(self, idx):
        return self.audio[idx], self.text[idx]
            

chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"\“]'           
dataset = CustomDataset("/home/sgowrira/Code-Mixed-Speech-Translation/Speech_Data/0001_Expand_To_Ten_Million/ChoppedAudio", "/home/sgowrira/Code-Mixed-Speech-Translation/Speech_Data/0001_Expand_To_Ten_Million/Hindi[100_].txt")
# data_loader = DataLoader(dataset, batch_size=1)
# for i, (audio, text) in enumerate(data_loader):
#     print(audio.shape)
#     print(text)
#     break

def predict(dataset):
    transcripts = []
    for i, (audio, text) in enumerate(dataset):
        #breakpoint()
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding="longest")
        input_values = inputs.input_values.to("cuda")
        attention_mask = inputs.attention_mask.to("cuda")
        
        with torch.no_grad():
            logits = model(input_values, attention_mask=attention_mask).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)
        transcripts.append(transcription[0])
    return transcripts


model = Wav2Vec2ForCTC.from_pretrained("theainerd/Wav2Vec2-large-xlsr-hindi").to("cuda")
processor = Wav2Vec2Processor.from_pretrained("theainerd/Wav2Vec2-large-xlsr-hindi")


transcripts = predict(dataset)
breakpoint()
wer_rate = load("wer")
wer_score = wer_rate.compute(predictions=transcripts, references=dataset.text)
print("WER:", wer_score)

