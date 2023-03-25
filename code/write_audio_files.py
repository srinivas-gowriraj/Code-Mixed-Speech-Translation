import os
from pathlib import Path

def fetch_audio_files(path):
    all_mp3 = list()
    folders = os.listdir(path)
    for fldr in folders:
        abspath = os.path.join(path, fldr)
        choppedpath = os.path.join(abspath, 'ChoppedAudio')
        mp3s = [os.path.join(choppedpath, f) for f in os.listdir(choppedpath)]
        all_mp3.extend(mp3s)
    
    return all_mp3



if __name__ == '__main__':
    print(fetch_audio_files("/home/medhnh/workhorse3/medi_dataset/mini_data"))
