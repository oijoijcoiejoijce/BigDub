import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Resize

from networks import DownBlock, AudioEncoder

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

        self.video_resize = Resize((256, 256))
        self.video_enc_conv = nn.ModuleList([
            nn.Conv2d(3 * T, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),

            DownBlock(64, 64),   # (B, 64, 128, 128)
            DownBlock(64, 64),  # (B, 128, 64, 64)
            DownBlock(64, 64),  # (B, 256, 32, 32)
            DownBlock(64, 128),  # (B, 256, 16, 16)
            DownBlock(128, 128),  # (B, 256, 8, 8)
            ]
        )
        self.video_enc_fc = nn.ModuleList([
            nn.Linear(128 * 8 * 8, 256),
            nn.LeakyReLU(),

            nn.Linear(256, 256),
            nn.LeakyReLU(),

            nn.Linear(256, 512)
        ])

        self.crit = nn.BCELoss()


    def audio_forward(self, audio):
        audio_enc = None
        if audio is not None:
            audio_enc = self.audio_enc(audio[:, None, :, :])
            audio_enc = audio_enc.reshape((audio_enc.shape[0], -1))
            audio_enc = F.relu(audio_enc)
            audio_enc = F.normalize(audio_enc, p=2, dim=-1)
        return audio_enc

    def param_forward(self, params):
        param_enc = None
        if params is not None:
            params = params.reshape((params.shape[0], -1))
            param_enc = self.param_enc(params)
            param_enc = F.relu(param_enc)
            param_enc = F.normalize(param_enc, p=2, dim=-1)
        return param_enc

    def video_forward(self, video):
        video_enc = None
        if video is not None:

            # Reshape and resize the video to have 256 x 256
            B, T, C, W, H = video.shape
            video = self.video_resize(video.reshape((-1, C, W, H)))
            video = video.reshape((B, T, -1, video.shape[-2], video.shape[-1]))

            # Stack the frames across the channel dimension
            video = video.reshape((video.shape[0], -1, video.shape[-2], video.shape[-1]))

            # Pass through the video encoder
            for layer in self.video_enc_conv:
                video = layer(video)

            video = video.reshape((video.shape[0], -1))
            for layer in self.video_enc_fc:
                video = layer(video)

            video = F.relu(video)
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

    def compute_loss(self, audio_enc_a=None, param_enc_a=None, video_enc_a=None,
                     audio_enc_b=None, param_enc_b=None, video_enc_b=None):

        loss = 0
        count = 0

        if audio_enc_a is not None:
            if audio_enc_b is not None:
                loss += self._compute_loss(audio_enc_a, audio_enc_b, is_same=False)
                count += 1
            if param_enc_b is not None:
                loss += self._compute_loss(audio_enc_a, param_enc_b, is_same=False)
                count += 1
            if video_enc_b is not None:
                loss += self._compute_loss(audio_enc_a, video_enc_b, is_same=False)
                count += 1
            if param_enc_a is not None:
                loss += self._compute_loss(audio_enc_a, param_enc_a, is_same=True)
                count += 1
            if video_enc_a is not None:
                loss += self._compute_loss(audio_enc_a, video_enc_a, is_same=True)
                count += 1

        if param_enc_a is not None:
            if audio_enc_b is not None:
                loss += self._compute_loss(param_enc_a, audio_enc_b, is_same=False)
                count += 1
            if param_enc_b is not None:
                loss += self._compute_loss(param_enc_a, param_enc_b, is_same=False)
                count += 1
            if video_enc_b is not None:
                loss += self._compute_loss(param_enc_a, video_enc_b, is_same=False)
                count += 1
            if video_enc_a is not None:
                loss += self._compute_loss(param_enc_a, video_enc_a, is_same=True)
                count += 1

        if video_enc_a is not None:
            if audio_enc_b is not None:
                loss += self._compute_loss(video_enc_a, audio_enc_b, is_same=False)
                count += 1
            if param_enc_b is not None:
                loss += self._compute_loss(video_enc_a, param_enc_b, is_same=False)
                count += 1
            if video_enc_b is not None:
                loss += self._compute_loss(video_enc_a, video_enc_b, is_same=False)
                count += 1

        if audio_enc_b is not None:
            if param_enc_b is not None:
                loss += self._compute_loss(audio_enc_b, param_enc_b, is_same=True)
                count += 1
            if video_enc_b is not None:
                loss += self._compute_loss(audio_enc_b, video_enc_b, is_same=True)
                count += 1

        if param_enc_b is not None:
            if video_enc_b is not None:
                loss += self._compute_loss(param_enc_b, video_enc_b, is_same=True)
                count += 1

        loss = loss / count
        return loss


