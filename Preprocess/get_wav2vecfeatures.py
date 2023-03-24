# Get MEL Spectrogram from Audio File and Save it to the Same Directory as the Video File
# Inspired by: https://github.com/Rudrabha/Wav2Lip/blob/master/audio.py
# TODO: Add hyperparameters to the command line arguments

import librosa
import librosa.display
import numpy as np
import os
from glob import glob
from tqdm import tqdm
import cv2
import torch

from Preprocess.wav2vec import Wav2Vec2Model, Wav2Vec2Config
from transformers import Wav2Vec2Processor


def preprocess_audio_single(in_path, out_path, model, processor):
    # Load audio
    audio, sr = librosa.load(in_path, sr=16000)
    audio_feature = np.squeeze(processor(audio, sampling_rate=16000).input_values)

    v_path = os.path.join(os.path.dirname(in_path), "video.mp4")
    cap = cv2.VideoCapture(v_path)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    hidden_states = model(torch.from_numpy(audio_feature[None]).float(), frame_num=n_frames).last_hidden_state
    np.save(out_path, hidden_states.detach().numpy()[0])


def preprocess_audio(in_root, model, processor):
    files = glob(os.path.join(in_root, '**', '*.wav'), recursive=True)

    for file in tqdm(files):
        out_path = os.path.join(os.path.dirname(file), 'wav2vec.npy')
        preprocess_audio_single(file, out_path, model, processor)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_path', type=str, required=True)

    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    # wav2vec 2.0 weights initialization
    model.feature_extractor._freeze_parameters()

    # audio_feature_map = nn.Linear(768, args.feature_dim)

    args = parser.parse_args()
    preprocess_audio(args.in_path, model, processor)
