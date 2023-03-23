import os

import cv2
import librosa
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

import torch
from torchvision.transforms import Compose, ToTensor, Resize
from enum import Enum


class DataTypes(Enum):
    Audio = 1
    MEL = 2
    IND_MEL = 3
    Params = 4
    Frames = 5
    ID = 6


class DubbingDataset(Dataset):
    def __init__(self, data_root, data_types, T=5, syncet=True, split='train',
                 fix_video=None, fix_ID=None, restrict_videos=None):
        super(DubbingDataset, self).__init__()
        self.data_root = data_root

        print(self.data_root)

        self.data = pd.read_csv(os.path.join(self.data_root, 'index.csv'))
        self.syncet = syncet
        self.data_types = data_types

        # Get the unique IDs
        self.ids = np.unique(np.array(self.data['face_ID']))
        self.n_ids = self.ids.shape[0]

        if split != 'all':
            self.data = self.data[self.data['split'] == split]

        if fix_ID is not None:
            self.data = self.data[self.data['face_ID'] == fix_ID]

        if restrict_videos is not None:
            self.data = self.data[self.data['v_path'].isin(restrict_videos)]

        # Add the path to the index
        """for i in range(len(self.data)):
            path = self.data.loc[i, 'v_path']
            if path[0] == '/':
                path = path[1:]
            self.data.loc[i, 'v_path'] = os.path.join(self.data_root, path)"""

        #self.len = self.data['length'].sum()
        self.len = len(self.data)  # This way makes validation per epoch work better using lightning
        self.T = T

        self.transform = Compose([ToTensor()])
        self.fix_video = fix_video

    def __len__(self):
        return self.len

    def sample_idxs(self, center_idx, valid_frames):
        idxs = np.arange(center_idx - self.T // 2, center_idx + self.T // 2 + 1)
        idxs = idxs.clip(0, max(valid_frames) - 1)
        return idxs

    def get_image_window(self, idxs, frames_dir):

        frames = []
        for idx in idxs:
            img = cv2.imread(os.path.join(frames_dir, f'{idx:05d}.png'), -1)

            if img.dtype == np.uint16:
                img = img.astype('float32') / (2 ** 16)

            if img is None:
                pass

            img = self.transform(img).float()
            frames.append(img[None])
        return torch.cat(frames)

    def get_video_window(self, idxs, video_path):
        frames = []
        cap = cv2.VideoCapture(video_path)
        for idx in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, img = cap.read()
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            if img.dtype == np.uint16:
                img = img.astype('float32') / (2 ** 16)

            if img is None:
                pass

            img = self.transform(img).float()
            frames.append(img[None])
        return torch.cat(frames)

    def __getitem__(self, item):
        while 1:
            # try:

                if self.fix_video is None:
                    v_idx = np.random.randint(len(self.data))
                    vid_root = self.data['v_path'].iloc[v_idx]
                    vid_root = os.path.join(self.data_root, vid_root)
                else:
                    vid_root = self.fix_video
                    for i in range(len(self.data['v_path'])):
                        if self.data['v_path'].iloc[i].replace('\\', '/') == vid_root.replace('\\', '/'):
                            v_idx = i
                            break

                length = self.data['length'].iloc[v_idx]
                if length <= 3 * self.T:
                    continue

                valid_frames = list(range(self.T, self.data['length'].iloc[v_idx] - self.T))

                # Create the return dict
                ret = {}

                # Target frame
                frame_idx = np.random.choice(valid_frames)
                frame_idxs = self.sample_idxs(frame_idx, valid_frames)

                ret['frame_idx'] = frame_idx
                ret['frame_idxs'] = frame_idxs

                # Get the frames
                if DataTypes.Frames in self.data_types:
                    frames = self.get_video_window(frame_idxs, os.path.join(vid_root, 'video.mp4'))
                    ret['frames'] = frames

                if DataTypes.Audio in self.data_types:
                    all_audio = librosa.load(os.path.join(vid_root, 'audio.wav'), sr=16000)[0]
                    scale = all_audio.shape[0] / length
                    start = int((min(frame_idxs) * scale))
                    audio_length = int(self.T * scale)
                    full_audio = all_audio[start:start+audio_length]
                    if full_audio.shape[0] != audio_length:
                        continue
                    ret['audio'] = full_audio

                # Get the MELS
                if DataTypes.MEL in self.data_types:
                    all_audio = np.load(os.path.join(vid_root, 'MEL.npy'), mmap_mode='r')

                    # scale = all_audio.shape[-1] / length

                    start, end = (frame_idx * (80 / 30)) - 8, (frame_idx * (80 / 30)) + 8
                    mel_idxs = np.arange(start, end).clip(0, all_audio.shape[-1] - 1).astype('int64')
                    full_MEL = all_audio[..., mel_idxs]

                    ind_mels = []
                    for idx in frame_idxs:
                        start, end = (idx * (80 / 30)) - 8, (idx * (80 / 30)) + 8
                        mel_idxs = np.arange(start, end).clip(0, all_audio.shape[-1] - 1).astype('int64')
                        mel = all_audio[..., mel_idxs][None]
                        ind_mels.append(mel)
                    ind_mels = np.concatenate(ind_mels)

                    ret['MEL'] = full_MEL
                    ret['ind_MEL'] = ind_mels

                # Get the ID
                if DataTypes.ID in self.data_types:
                    ID = self.data['face_ID'].iloc[v_idx]
                    ret['ID'] = ID

                if DataTypes.Params in self.data_types:
                    all_params = dict(np.load(os.path.join(vid_root, 'params.npz')))
                    params = {p: all_params[p][frame_idxs] for p in all_params.keys()}
                    ret['params'] = params

                if not self.syncet:
                    return ret

                # If in syncet mode, select a random frame from the same video for parameters and audio
                # Remove +- 5 frames from the valid frames
                valid_frames = [f for f in valid_frames if f not in range(frame_idx - self.T, frame_idx + self.T)]

                # Target frame
                other_frame_idx = np.random.choice(valid_frames)
                other_frame_idxs = self.sample_idxs(other_frame_idx, valid_frames)

                ret['other_frame_idx'] = other_frame_idx
                ret['other_frame_idxs'] = other_frame_idxs

                # Get the frames
                if DataTypes.Frames in self.data_types:
                    other_frames  = self.get_video_window(other_frame_idxs, os.path.join(vid_root, 'video.mp4'))
                    ret['other_frames'] = other_frames

                if DataTypes.Audio in self.data_types:
                    other_full_audio = all_audio[min(other_frame_idxs) * 16000: max(other_frame_idxs) * 16000]
                    ret['other_audio'] = other_full_audio

                # Get the MELS
                if DataTypes.MEL in self.data_types:
                    start, end = (other_frame_idx * (80 / 30)) - 8, (other_frame_idx * (80 / 30)) + 8
                    mel_idxs = np.arange(start, end).clip(0, all_audio.shape[-1] - 1).astype('int64')
                    other_MEL = all_audio[..., mel_idxs]

                    ind_mels = []
                    for idx in other_frame_idxs:
                        start, end = (idx * (80 / 30)) - 8, (idx * (80 / 30)) + 8
                        mel_idxs = np.arange(start, end).clip(0, all_audio.shape[-1] - 1).astype('int64')
                        mel = all_audio[..., mel_idxs][None]
                        ind_mels.append(mel)
                    ind_mels = np.concatenate(ind_mels)

                    ret['other_MEL'] = other_MEL
                    ret['other_ind_MEL'] = ind_mels

                # Get the params
                if DataTypes.Params in self.data_types:
                    other_params = {p: all_params[p][other_frame_idxs] for p in all_params.keys()}
                    ret['other_params'] = other_params

                # Return the dict
                return ret

            #except Exception as e:
            #    print(e)
            #    continue

    def get_video_generator(self, specific_video=None):
        """Returns a generator that yields a dict of data for each frame in the video

            Args:
                specific_video (str): If not None, will only yield frames from this video # TODO: Implement this
        """
        v_idx = [os.path.basename(x) for x in self.data['v_path'].unique()].index(specific_video) if specific_video is not None else None

        if v_idx is None:
            v_idx = np.random.randint(len(self.data))

        vid_root = self.data['v_path'].iloc[v_idx]
        vid_root = os.path.join(self.data_root, vid_root)
        length = self.data['length'].iloc[v_idx]

        valid_frames = list(range(self.T, length - self.T))

        def generator(frame_idx):

            # Create the return dict
            ret = {}

            # Target frame
            # frame_idx = np.random.choice(valid_frames)
            frame_idxs = self.sample_idxs(frame_idx, valid_frames)

            ret['frame_idx'] = frame_idx
            ret['frame_idxs'] = frame_idxs

            # Get the frames
            if DataTypes.Frames in self.data_types:
                frames = self.get_video_window(frame_idxs, os.path.join(vid_root, 'video.mp4'))
                ret['frames'] = frames

            if DataTypes.Audio in self.data_types:
                all_audio = librosa.load(os.path.join(vid_root, 'audio.wav'), sr=16000)[0]
                scale = all_audio.shape[0] / length
                start = int((min(frame_idxs) * scale))
                audio_length = int(self.T * scale)
                full_audio = all_audio[start:start+audio_length]
                ret['audio'] = full_audio

            # Get the MELS
            if DataTypes.MEL in self.data_types:
                all_audio = np.load(os.path.join(vid_root, 'MEL.npy'), mmap_mode='r')

                start, end = (frame_idx * (80 / 30)) - 8, (frame_idx * (80 / 30)) + 8
                mel_idxs = np.arange(start, end).clip(0, all_audio.shape[-1] - 1).astype('int64')
                full_MEL = all_audio[..., mel_idxs]

                ind_mels = []
                for idx in frame_idxs:
                    start, end = (idx * (80 / 30)) - 8, (idx * (80 / 30)) + 8
                    mel_idxs = np.arange(start, end).clip(0, all_audio.shape[-1] - 1).astype('int64')
                    mel = all_audio[..., mel_idxs][None]
                    ind_mels.append(mel)
                ind_mels = np.concatenate(ind_mels)

                ret['MEL'] = full_MEL
                ret['ind_MEL'] = ind_mels

            # Get the ID
            if DataTypes.ID in self.data_types:
                ID = self.data['face_ID'].iloc[v_idx]
                ret['ID'] = ID

            if DataTypes.Params in self.data_types:
                all_params = dict(np.load(os.path.join(vid_root, 'params.npz')))
                params = {p: all_params[p][frame_idxs] for p in all_params.keys()}
                ret['params'] = params

            # Return the dict
            return ret
        return generator, length

if __name__ == '__main__':

    dataset = DubbingDataset(
        'C:/Users/jacks/Documents/Data/DubbingForExtras/v3/',
        data_types=[DataTypes.Frames, DataTypes.MEL, DataTypes.Params])

    gen, n_frames = dataset.get_video_generator()
    import cv2

    for i in range(n_frames):
        frames = gen(i)['frames']
        cv2.imshow('Frame', frames[0].permute((1, 2, 0)).detach().cpu().numpy())
        cv2.waitKey(1)

