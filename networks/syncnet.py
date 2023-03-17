import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Resize
import mediapipe as mp
import numpy as np

from networks import DownBlock, AudioEncoder


class Conv2d(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
                            nn.Conv2d(cin, cout, kernel_size, stride, padding),
                            nn.BatchNorm2d(cout)
                            )
        self.act = nn.ReLU()
        self.residual = residual

    def forward(self, x):
        out = self.conv_block(x)
        if self.residual:
            out += x
        return self.act(out)

class TripleSyncnet(nn.Module):
    """SyncNet with 3 inputs: audio, video, and parameters.
   """
    def __init__(self, n_params, T=5):

        super(TripleSyncnet, self).__init__()
        self.audio_enc = AudioEncoder()

        self.param_enc = nn.Sequential(
            nn.Linear(n_params * T, 512),
            nn.LeakyReLU(),

            nn.Linear(512, 512),
            nn.LeakyReLU(),

            nn.Linear(512, 512),
            nn.LeakyReLU(),

            nn.Linear(512, 512)
        )

        self.video_resize = Resize((48, 96))
        self.video_enc_conv = nn.ModuleList([
            nn.Conv2d(3 * T, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),

            DownBlock(64, 64),   # (B, 64, 64, 64)
            DownBlock(64, 64),  # (B, 256, 32, 32)
            DownBlock(64, 128),  # (B, 256, 16, 16)
            DownBlock(128, 128),  # (B, 256, 8, 8)
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
            ]
        )
        self.video_enc_fc = nn.ModuleList([
            nn.Linear(128 * 8 * 8, 256),
            nn.LeakyReLU(),

            nn.Linear(256, 256),
            nn.LeakyReLU(),

            nn.Linear(256, 512)
        ])

        self.face_encoder = nn.ModuleList([
            Conv2d(15, 32, kernel_size=(7, 7), stride=1, padding=3),

            Conv2d(32, 64, kernel_size=5, stride=(1, 2), padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0)])

        self.crit = nn.BCELoss()
        self.triplet = nn.TripletMarginLoss(margin=0.2)
        self.det = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, static_image_mode=True, refine_landmarks=False)
        self.lmk_idxs = [212, 432, 18, 164]

    def get_mouth_bb(self, frame):

        frame_np = (frame.permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
        lmks = self.det.process(frame_np)
        if lmks.multi_face_landmarks is None:
            return -1

        lmks = lmks.multi_face_landmarks[0]
        lmks = lmks.landmark
        lmks = [[int(lmk.x * frame.shape[2]), int(lmk.y * frame.shape[1])] for lmk in lmks]
        lmks = [lmks[idx] for idx in self.lmk_idxs]
        bb = [min([lmk[0] for lmk in lmks]), min([lmk[1] for lmk in lmks]), max([lmk[0] for lmk in lmks]),
              max([lmk[1] for lmk in lmks])]

        return bb

    def crop_video_to_mouth(self, video):
        B, T, C, W, H = video.shape
        video = video.reshape((-1, C, W, H))
        bb = -1
        crops = []
        for i in range(video.shape[0]):
            possible_bb = self.get_mouth_bb(video[i])
            if possible_bb != -1:
                bb = possible_bb
            cropped_frame = video[i, :, bb[1]:bb[3], bb[0]:bb[2]]
            cropped_frame = self.video_resize(cropped_frame[None, :, :, :])
            crops.append(cropped_frame)
        crops = torch.cat(crops, dim=0)
        crops = crops.reshape((B, T, C, crops.shape[-2], crops.shape[-1]))
        return crops

    def audio_forward(self, audio):
        audio_enc = None
        if audio is not None:
            audio_enc = self.audio_enc(audio[:, None, :, :])
            audio_enc = audio_enc.reshape((audio_enc.shape[0], -1))
            #audio_enc = F.relu(audio_enc)
            audio_enc = torch.sigmoid(audio_enc)
            audio_enc = F.normalize(audio_enc, p=2, dim=-1)
        return audio_enc

    def param_forward(self, params):
        param_enc = None
        if params is not None:
            params = params.reshape((params.shape[0], -1))
            param_enc = self.param_enc(params)
            #param_enc = F.relu(param_enc)
            param_enc = torch.sigmoid(param_enc)
            param_enc = F.normalize(param_enc, p=2, dim=-1)
        return param_enc

    def video_forward(self, video):
        video_enc = None
        if video is not None:

            video = self.crop_video_to_mouth(video)

            # Reshape and resize the video to have 128, 128
            B, T, C, W, H = video.shape

            # video = self.video_resize(video.reshape((-1, C, W, H)))
            video = video.reshape((B, T, -1, video.shape[-2], video.shape[-1]))

            # Stack the frames across the channel dimension
            video = video.reshape((video.shape[0], -1, video.shape[-2], video.shape[-1]))

            # Pass through the video encoder
            #for layer in self.video_enc_conv:
            for layer in self.face_encoder:
                video = layer(video)

            video = video.reshape((video.shape[0], -1))
            #for layer in self.video_enc_fc:
            #    video = layer(video)

            video_enc = video

            #video = F.relu(video)
            video = torch.sigmoid(video)
            video_enc = F.normalize(video, p=2, dim=-1)
        return video_enc


    # @profile
    def forward(self, audio=None, params=None, video=None):
        # Audio = (B, 80, 16), params = (B, T, C), video = (B, T, 3, 256, 256)

        audio_enc = self.audio_forward(audio)
        param_enc = self.param_forward(params)
        video_enc = self.video_forward(video)

        return audio_enc, param_enc, video_enc

    def _compute_loss(self, enc_a, enc_b, is_same):

        similarity = F.cosine_similarity(enc_a, enc_b, dim=-1).clip(0, 1)
        if is_same:
            target = torch.ones_like(similarity)
        else:
            target = torch.zeros_like(similarity)

        loss = self.crit(similarity, target)
        return loss

    def compute_triplet_loss(self, audio_enc_a, param_enc_a, video_enc_a,
                        audio_enc_b, param_enc_b, video_enc_b):
        """Compute the triplet loss for the 3 encodings"""

        loss = self.triplet(audio_enc_a, video_enc_a, audio_enc_b)
        loss += self.triplet(param_enc_b, video_enc_b, param_enc_a)
        loss += self.triplet(video_enc_a, param_enc_a, video_enc_b)
        return loss

    def compute_loss(self, audio_enc_a=None, param_enc_a=None, video_enc_a=None,
                     audio_enc_b=None, param_enc_b=None, video_enc_b=None):

        """total_loss = 0
        count = 0
        loss_dict = {}

        if audio_enc_a is not None:
            if audio_enc_b is not None:
                audio_audio_different = self._compute_loss(audio_enc_a, audio_enc_b, is_same=False)
                total_loss += audio_audio_different
                loss_dict['audio_audio_different'] = audio_audio_different.item()
                count += 1
            if param_enc_b is not None:
                audio_param_different = self._compute_loss(audio_enc_a, param_enc_b, is_same=False)
                total_loss += audio_param_different
                loss_dict['audio_param_different'] = audio_param_different.item()
                count += 1
            if video_enc_b is not None:
                audio_video_different = self._compute_loss(audio_enc_a, video_enc_b, is_same=False)
                total_loss += audio_video_different
                loss_dict['audio_video_different'] = audio_video_different.item()
                count += 1
            if param_enc_a is not None:
                audio_param_same = self._compute_loss(audio_enc_a, param_enc_a, is_same=True)
                total_loss += audio_param_same
                loss_dict['audio_param_same'] = audio_param_same.item()
                count += 1
            if video_enc_a is not None:
                audio_video_same = self._compute_loss(audio_enc_a, video_enc_a, is_same=True)
                total_loss += audio_video_same
                loss_dict['audio_video_same'] = audio_video_same.item()
                count += 1

        if param_enc_a is not None:
            if audio_enc_b is not None:
                param_audio_different = self._compute_loss(param_enc_a, audio_enc_b, is_same=False)
                total_loss += param_audio_different
                loss_dict['param_audio_different'] = param_audio_different.item()
                count += 1
            if param_enc_b is not None:
                param_param_different = self._compute_loss(param_enc_a, param_enc_b, is_same=False)
                total_loss += param_param_different
                loss_dict['param_param_different'] = param_param_different.item()
                count += 1
            if video_enc_b is not None:
                param_video_different = self._compute_loss(param_enc_a, video_enc_b, is_same=False)
                total_loss += param_video_different
                loss_dict['param_video_different'] = param_video_different.item()
                count += 1
            if video_enc_a is not None:
                param_video_same = self._compute_loss(param_enc_a, video_enc_a, is_same=True)
                total_loss += param_video_same
                loss_dict['param_video_same'] = param_video_same.item()
                count += 1

        if video_enc_a is not None:
            if audio_enc_b is not None:
                video_audio_different = self._compute_loss(video_enc_a, audio_enc_b, is_same=False)
                total_loss += video_audio_different
                loss_dict['video_audio_different'] = video_audio_different.item()
                count += 1
            if param_enc_b is not None:
                video_param_different = self._compute_loss(video_enc_a, param_enc_b, is_same=False)
                total_loss += video_param_different
                loss_dict['video_param_different'] = video_param_different.item()
                count += 1
            if video_enc_b is not None:
                video_video_different = self._compute_loss(video_enc_a, video_enc_b, is_same=False)
                total_loss += video_video_different
                loss_dict['video_video_different'] = video_video_different.item()
                count += 1

        if audio_enc_b is not None:
            if param_enc_b is not None:
                audio_param_same = self._compute_loss(audio_enc_b, param_enc_b, is_same=True)
                total_loss += audio_param_same
                loss_dict['audio_param_same'] = audio_param_same.item()
                count += 1
            if video_enc_b is not None:
                audio_video_same = self._compute_loss(audio_enc_b, video_enc_b, is_same=True)
                total_loss += audio_video_same
                loss_dict['audio_video_same'] = audio_video_same.item()
                count += 1

        if param_enc_b is not None:
            if video_enc_b is not None:
                param_video_same = self._compute_loss(param_enc_b, video_enc_b, is_same=True)
                total_loss += param_video_same
                loss_dict['param_video_same'] = param_video_same.item()
                count += 1

        loss = total_loss / count
        """

        loss_same = self._compute_loss(video_enc_a, param_enc_a, is_same=True)
        loss_different = self._compute_loss(video_enc_a, param_enc_b, is_same=False)

        loss = (loss_same + loss_different) / 2

        return loss