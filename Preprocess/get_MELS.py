# Get MEL Spectrogram from Audio File and Save it to the Same Directory as the Video File
# Inspired by: https://github.com/Rudrabha/Wav2Lip/blob/master/audio.py
# TODO: Add hyperparameters to the command line arguments

import librosa
import librosa.display
import numpy as np
import os
from glob import glob
from tqdm import tqdm


def preprocess_audio_single(in_path, out_path):
    audio, sr = librosa.load(in_path, sr=16000)

    # Scale the audio
    audio = audio / np.max(np.abs(audio))
    audio = audio * 0.95

    # Preemphasis
    audio = librosa.effects.preemphasis(audio)

    # Get MEL Spectrogram
    S = librosa.feature.melspectrogram(y=audio, sr=sr,
                                       n_fft=512, hop_length=200, n_mels=80)

    S = librosa.amplitude_to_db(S, ref=20)
    S = np.clip((2 * 4) * ((S - -100) / (- -100)) - 4,
            -4, 4)

    np.save(out_path, S)


def preprocess_audio(in_root):
    files = glob(os.path.join(in_root, '**', '*.wav'), recursive=True)

    for file in tqdm(files):
        out_path = os.path.join(os.path.dirname(file), 'MEL.npy')
        preprocess_audio_single(file, out_path)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_path', type=str, required=True)

    args = parser.parse_args()
    preprocess_audio(args.in_path)
