import os
import torchaudio
import speechbrain as sb
from speechbrain.pretrained import EncoderClassifier
from write_audio_files import fetch_audio_files
from params import get_params


def detect_language(audio_file):
        waveform, sample_rate = torchaudio.load(audio_file)
        #print(waveform.shape)
        frame_length = 2 * sample_rate
        frames = waveform.unfold(1, frame_length, frame_length)

        #print(frames.shape)
        language_id = EncoderClassifier.from_hparams(source="speechbrain/lang-id-voxlingua107-ecapa", savedir="tmp")
        
        frame_languages = list()
        for i, frame in enumerate(frames[1]):
            language = language_id.classify_batch(frame)
            frame_languages.append(language[3])
        return frame_languages


if __name__ == '__main__':
    args = get_params()

    files = fetch_audio_files(args.data_path)


    for audio_file in files[:4]:
        file_name = os.path.basename(audio_file)
        detected_languages = detect_language(audio_file)
        print(file_name, detected_languages)

    
