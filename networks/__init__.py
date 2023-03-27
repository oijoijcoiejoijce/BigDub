# TODO: Seperate the networks into different files

import torch
import torch.nn as nn
import torch.nn.functional as F

# from line_profiler_pycharm import profile

ACTIVATIONS = {
    'relu': nn.ReLU,
    'leaky_relu': nn.LeakyReLU,
    'selu': nn.SELU,
    'silu': nn.SiLU
}


class WSLinear(nn.Module):
    def __init__(
            self, in_features, out_features
    ):
        super(WSLinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.scale = (2 / in_features) ** 0.5
        self.bias = self.linear.bias
        self.linear.bias = None

        nn.init.normal_(self.linear.weight)
        nn.init.zeros_(self.bias)

    # @profile
    def forward(self, x):
        return self.linear(x * self.scale) + self.bias


class AdaIN(nn.Module):

    def __init__(self, style_dim, n_feat):
        super(AdaIN, self).__init__()

        self.norm = nn.InstanceNorm2d(n_feat)
        self.mu = WSLinear(style_dim, n_feat)
        self.sigma = WSLinear(style_dim, n_feat)

    # @profile
    def forward(self, x, style):
        x = self.norm(x)
        mu = self.mu(style).unsqueeze(2).unsqueeze(3)
        sigma = self.sigma(style).unsqueeze(2).unsqueeze(3)

        #return (x * sigma) + mu
        return x

class BasicBlock(nn.Module):

    def __init__(self, in_c, out_c, h_c=None, activation='silu', use_norm=True):
        super(BasicBlock, self).__init__()
        if h_c is None:
            h_c = out_c

        self.conv1 = nn.Conv2d(in_c, h_c, kernel_size=3, padding=1, bias=True, padding_mode='reflect')
        self.act1 = ACTIVATIONS[activation]()

        self.conv2 = nn.Conv2d(h_c, out_c, kernel_size=3, padding=1, bias=True, padding_mode='reflect')
        self.act2 = ACTIVATIONS[activation]()

        if use_norm:
            self.norm1 = nn.InstanceNorm2d(h_c)
            self.norm2 = nn.InstanceNorm2d(out_c)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()

    # @profile
    def forward(self, x):

        x = self.conv1(x)
        x = self.act1(x)
        x = self.norm1(x)

        x = self.conv2(x)
        x = self.act2(x)
        x = self.norm2(x)

        return x


class ResBlock(nn.Module):

    def __init__(self, in_c, out_c, h_c=None, activation='silu', use_norm=True):
        super(ResBlock, self).__init__()
        if h_c is None:
            h_c = out_c

        self.conv1 = nn.Conv2d(in_c, h_c, kernel_size=3, padding=1, bias=True, padding_mode='reflect')
        self.act1 = ACTIVATIONS[activation]()

        self.conv2 = nn.Conv2d(h_c, out_c, kernel_size=3, padding=1, bias=True, padding_mode='reflect')
        self.act2 = ACTIVATIONS[activation]()

        if use_norm:
            self.norm1 = nn.InstanceNorm2d(h_c)
            self.norm2 = nn.InstanceNorm2d(out_c)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()

    # @profile
    def forward(self, x):

        x0 = self.conv1(x)
        x0 = self.act1(x0)
        x0 = self.norm1(x0)

        x1 = self.conv2(x0)
        x1 = self.act2(x1)
        x1 = self.norm2(x1)

        return x + x1


class DownBlock(nn.Module):

    def __init__(self, in_c, out_c, downscale=2):
        super(DownBlock, self).__init__()

        self.down = nn.Conv2d(in_c, in_c, kernel_size=3, padding=1, padding_mode='reflect', stride=downscale)
        self.norm = nn.InstanceNorm2d(in_c)
        self.net = BasicBlock(in_c, out_c)

    # @profile
    def forward(self, x):
        x = self.down(x)
        x = self.norm(x)
        return self.net(x)


class UpBlock(nn.Module):

    def __init__(self, in_c, out_c, downscale=2, use_norm=True):
        super(UpBlock, self).__init__()

        self.net = BasicBlock(in_c * 2, out_c, use_norm=use_norm)
        self.up = nn.UpsamplingBilinear2d(scale_factor=downscale)

    # @profile
    def forward(self, x, x_skip):
        x = torch.cat((x_skip, x), dim=1)
        x = self.net(x)
        return self.up(x)


class UnetAdaIN(nn.Module):

    def __init__(self, in_c, out_c, n_blocks=4, downscale=2, init_features=32, cond_c=256):
        super(UnetAdaIN, self).__init__()

        self.in_conv = nn.Conv2d(in_c, init_features, kernel_size=7, padding=3, padding_mode='reflect')
        downs = []
        n_feat = init_features
        for block in range(n_blocks):
            downs.append(DownBlock(n_feat, n_feat * 2, downscale=downscale))
            n_feat *= 2
        self.downs = nn.ModuleList(downs)

        self.bottleneck = nn.ModuleList([ResBlock(n_feat, n_feat), ResBlock(n_feat, n_feat), ResBlock(n_feat, n_feat)])

        ups, adas = [], []
        n_cond = n_feat
        for block in range(n_blocks):
            ups.append(UpBlock(n_feat, n_feat // 2, downscale=downscale, use_norm=False))
            adas.append(AdaIN(n_cond, n_feat // 2))
            n_feat = n_feat // 2

        self.ups = nn.ModuleList(ups)
        self.adas = nn.ModuleList(adas)

        self.final_block = BasicBlock(n_feat, n_feat)
        self.out_conv = nn.Conv2d(n_feat, out_c, kernel_size=1, padding=0)

        self.mapping = nn.Sequential(
            nn.Linear(cond_c, cond_c),
            nn.LeakyReLU(),

            nn.Linear(cond_c, cond_c),
            nn.LeakyReLU(),

            nn.Linear(cond_c, cond_c),
            nn.LeakyReLU()
        )

    # @profile
    def forward(self, x, x_ref):

        # x_ref = self.mapping(x_ref)

        x0 = self.in_conv(x)
        x_conds = [x0]
        x = x0

        for down in self.downs:
            x = down(x)
            x_conds.append(x)

        # for i, layer in enumerate(self.bottleneck):
        #    x = layer(x)

        for i, up in enumerate(self.ups):
            x_skip = x_conds.pop(-1)
            x = up(x, x_skip)
            x = self.adas[i](x, x_ref.squeeze())

        x = self.final_block(x)
        # return torch.sigmoid(self.out_conv(x))
        return self.out_conv(x)


class AudioEncoder(nn.Module):

    def __init__(self):
        super(AudioEncoder, self).__init__()
        self.net = nn.ModuleList([
            BasicBlock(1, 32),
            BasicBlock(32, 32),
            BasicBlock(32, 32),

            nn.Conv2d(32, 64, kernel_size=3, stride=(3, 1), padding=1),
            BasicBlock(64, 64),
            BasicBlock(64, 64),

            nn.Conv2d(64, 128, kernel_size=3, stride=3, padding=1),
            BasicBlock(128, 128),
            BasicBlock(128, 128),

            nn.Conv2d(128, 256, kernel_size=3, stride=(3, 2), padding=1),
            BasicBlock(256, 256),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0)
        ])

    # @profile
    def forward(self, x):
        for layer in self.net:
            x = layer(x)
        return x


class Discriminator(nn.Module):

    def __init__(self, in_c, init_h_c=64, n_layers=5, image_size=256):
        super(Discriminator, self).__init__()

        self.in_conv = nn.Conv2d(in_c, init_h_c, padding=1, kernel_size=3, padding_mode='reflect')
        self.image_size = image_size
        self.final_size = image_size // (2 ** n_layers)
        downs = []
        nf = init_h_c
        for layer in range(n_layers):
            downs.append(DownBlock(nf, nf))
            # nf *= 2
        self.downs = nn.ModuleList(downs)

        # TODO: Add attention layer

        self.out_conv = nn.Sequential(
            nn.Conv2d(nf, nf, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.LeakyReLU(0.2, True),
            nn.InstanceNorm2d(nf),
            nn.Conv2d(nf, 1, kernel_size=1)
        )

        #self.out_fc = nn.Sequential(
        #    nn.Linear(nf * self.final_size * self.final_size, 512),
        #    nn.LeakyReLU(0.2, True),
        #    nn.Linear(512, 1)
        #)


    # @profile
    def forward(self, x):
        x = self.in_conv(x)
        for down in self.downs:
            x = down(x)
        return self.out_conv(x)


class Audio2Parameter(nn.Module):

    def __init__(self, n_params, cond_c=None):
        super(Audio2Parameter, self).__init__()
        self.audio_enc = AudioEncoder()

        self.n_in = 512
        self.use_cond = False
        if cond_c is not None:
            self.n_in += cond_c
            self.use_cond = True

        self.MLP = nn.Sequential(
            nn.Linear(self.n_in, 512),
            nn.Linear(self.n_in, 512),
            nn.LeakyReLU(),

            nn.Linear(512, 256),
            nn.LeakyReLU(),

            nn.Linear(256, n_params)
        )

    # @profile
    def forward(self, audio, cond=None):
        # audio = (B, 80, 16), cond = None or (B, C)
        # out = (B, N)
        if self.use_cond and cond is None:
            raise ValueError('Condition is specified but not provided')

        x = self.audio_enc(audio).reshape((audio.shpae[0], 512))
        if cond is not None:
            x = torch.cat((x, cond), dim=-1)
        return self.MLP(x)


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

            DownBlock(64, 64),  # (B, 64, 128, 128)
            DownBlock(64, 64),  # (B, 128, 64, 64)
            DownBlock(64, 64),  # (B, 256, 32, 32)
            DownBlock(64, 128),  # (B, 256, 16, 16)
            DownBlock(128, 128),  # (B, 256, 8, 8)
            DownBlock(128, 128),  # (B, 256, 4, 4)
            DownBlock(128, 256),  # (B, 256, 2, 2)
            DownBlock(256, 512),  # (B, 512, 1, 1)
        )

    # @profile
    def forward(self, audio=None, params=None, video=None):
        # Audio = (B, 80, 16), params = (B, T, C), video = (B, 3, 256, 256)

        audio_enc = None
        if audio is not None:
            audio_enc = self.audio_enc(audio)
            audio_enc = F.normalize(audio_enc, dim=-1)

        param_enc = None
        if params is not None:
            param_enc = self.param_enc(params)
            param_enc = F.normalize(param_enc, dim=-1)

        video_enc = None
        if video is not None:
            video = video.reshape((video.shape[0], -1, video.shape[2], video.shape[3]))
            video = self.video_enc(video)
            video_enc = F.normalize(video, dim=-1)

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
