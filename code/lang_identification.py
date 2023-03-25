import os
import re
import torch
import torchaudio
import speechbrain as sb
from speechbrain.pretrained import EncoderClassifier
from write_audio_files import fetch_audio_files
from params import get_params

language_id = EncoderClassifier.from_hparams(source="speechbrain/lang-id-voxlingua107-ecapa", savedir="tmp")

def detect_language_spans(frame_lang_list):
    last_lang = frame_lang_list[0]
    current_lang = None
    current_start = 0
    spans = list()
    length = len(frame_lang_list)
    i = 1
    while i < length:
        current_lang = frame_lang_list[i]
        if current_lang != last_lang:
            span = (current_start, i, last_lang)
            spans.append(span)
            current_start = i
            last_lang = current_lang
        i += 1
    spans.append((current_start, length, current_lang))
    return spans



def detect_language(audio_file, languages=['en: English', 'sa: Sanskrit', 'bn: Bengali']):
        dirname, basename = os.path.split(audio_file)
        transcript_dirname, _ = os.path.split(dirname)

        str_index = re.findall('(\d+)\.mp3', basename)
        index = int(str_index[0])

        with open(f"{transcript_dirname}/English[100_].txt") as f:
             text = f.readlines()
             text = text[index - 1].strip("\n").split("_")[-1]

        waveform, sample_rate = torchaudio.load(audio_file)
        selected_labels = torch.as_tensor([language_id.hparams.label_encoder.lab2ind[lang] for lang in languages])
        #print(waveform.shape)
        frame_length = 2 * sample_rate
        frames = waveform.unfold(1, frame_length, frame_length)

        # print(frames.shape)
        
        frame_languages = list()
        for i, frame in enumerate(frames[1]):
            output = language_id.classify_batch(frame)
            probs = output[0][:, selected_labels]  # batch_size x #selected_labels
            predicted_idx = torch.argmax(probs, dim=1)  # batch_size x #selected_labels
            predicted_idx_remapped = torch.as_tensor([selected_labels[predicted_idx.item()]])
            predicted_label = language_id.hparams.label_encoder.decode_torch(predicted_idx_remapped)
            frame_languages.append(predicted_label)
        
        spans = detect_language_spans(frame_languages)
        split_audios = [frames[:, sp[0]:sp[1], :] for sp in spans]
        return frame_languages, text, split_audios


if __name__ == '__main__':
    args = get_params()

    files = fetch_audio_files(args.data_path)


    for audio_file in files[:4]:
        file_name = os.path.basename(audio_file)
        detected_languages = detect_language(audio_file)
        print(file_name, detected_languages)
        # print(detect_language_spans(detected_languages))
    
