import torch
import torch.nn as nn
import torch.nn.functional as F

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

        self.video_enc = nn.Sequential(
            nn.Conv2d(3 * T, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),

            DownBlock(64, 64),   # (B, 64, 128, 128)
            DownBlock(64, 64),  # (B, 128, 64, 64)
            DownBlock(64, 64),  # (B, 256, 32, 32)
            DownBlock(64, 128),  # (B, 256, 16, 16)
            DownBlock(128, 128),  # (B, 256, 8, 8)
            DownBlock(128, 128),  # (B, 256, 4, 4)
            DownBlock(128, 256),  # (B, 256, 2, 2)
            DownBlock(256, 512),  # (B, 512, 1, 1)
        )

    def audio_forward(self, audio):
        audio_enc = None
        if audio is not None:
            audio_enc = self.audio_enc(audio)
            audio_enc = F.normalize(audio_enc, dim=-1)
        return audio_enc

    def param_forward(self, params):
        param_enc = None
        if params is not None:
            params = params.reshape((params.shape[0], -1))
            param_enc = self.param_enc(params)
            param_enc = F.normalize(param_enc, dim=-1)
        return param_enc

    def video_forward(self, video):
        video_enc = None
        if video is not None:
            video = video.reshape((video.shape[0], -1, video.shape[-2], video.shape[-1]))
            video = self.video_enc(video)
            video_enc = F.normalize(video, dim=-1)
        return video_enc


    # @profile
    def forward(self, audio=None, params=None, video=None):
        # Audio = (B, 80, 16), params = (B, T, C), video = (B, T, 3, 256, 256)

        audio_enc = self.audio_forward(audio)
        param_enc = self.param_forward(params)
        video_enc = self.video_forward(video)

        return audio_enc, param_enc, video_enc

    def _compute_loss(self, enc_a, enc_b, is_same):

        similariy = F.cosine_similarity(enc_a, enc_b, dim=-1)
        if is_same:
            target = torch.ones_like(similariy)
        else:
            target = torch.zeros_like(similariy)

        loss = F.binary_cross_entropy(similariy, target)
        return loss

    def compute_loss(self, audio_enc_a=None, param_enc_a=None, video_enc_a=None,
                     audio_enc_b=None, param_enc_b=None, video_enc_b=None):

        loss = 0

        if audio_enc_a is not None:
            if audio_enc_b is not None:
                loss += self._compute_loss(audio_enc_a, audio_enc_b, is_same=False)
            if param_enc_b is not None:
                loss += self._compute_loss(audio_enc_a, param_enc_b, is_same=False)
            if video_enc_b is not None:
                loss += self._compute_loss(audio_enc_a, video_enc_b, is_same=False)
            if param_enc_a is not None:
                loss += self._compute_loss(audio_enc_a, param_enc_a, is_same=True)
            if video_enc_a is not None:
                loss += self._compute_loss(audio_enc_a, video_enc_a, is_same=True)

        if param_enc_a is not None:
            if audio_enc_b is not None:
                loss += self._compute_loss(param_enc_a, audio_enc_b, is_same=False)
            if param_enc_b is not None:
                loss += self._compute_loss(param_enc_a, param_enc_b, is_same=False)
            if video_enc_b is not None:
                loss += self._compute_loss(param_enc_a, video_enc_b, is_same=False)
            if video_enc_a is not None:
                loss += self._compute_loss(param_enc_a, video_enc_a, is_same=True)

        if video_enc_a is not None:
            if audio_enc_b is not None:
                loss += self._compute_loss(video_enc_a, audio_enc_b, is_same=False)
            if param_enc_b is not None:
                loss += self._compute_loss(video_enc_a, param_enc_b, is_same=False)
            if video_enc_b is not None:
                loss += self._compute_loss(video_enc_a, video_enc_b, is_same=False)

        if audio_enc_b is not None:
            if param_enc_b is not None:
                loss += self._compute_loss(audio_enc_b, param_enc_b, is_same=True)
            if video_enc_b is not None:
                loss += self._compute_loss(audio_enc_b, video_enc_b, is_same=True)

        if param_enc_b is not None:
            if video_enc_b is not None:
                loss += self._compute_loss(param_enc_b, video_enc_b, is_same=True)

        return loss


