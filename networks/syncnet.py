import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Resize
import mediapipe as mp
import numpy as np

from networks import DownBlock, AudioEncoder
import trimesh

from EMOCA_lite.DecaFLAME import FLAME, FLAMETex
from EMOCA_lite.Renderer import ComaMeshRenderer

import os
from omegaconf import OmegaConf

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

class ParamsToImage(nn.Module):

    def __init__(self, config):
        super().__init__()

        emoca_checkpoint = config.get("Paths", "emoca_checkpoint")
        emoca_config = config.get("Paths", "emoca_config")
        flame_assets = config.get("Paths", "flame_assets")

        with open(emoca_config, "r") as f:
            conf = OmegaConf.load(f)
        conf = conf.detail

        model_cfg = conf.model
        model_cfg.mode = "coarse"
        model_cfg.resume_training = False

        for k in ["topology_path", "fixed_displacement_path", "flame_model_path", "flame_lmk_embedding_path",
                  "flame_mediapipe_lmk_embedding_path", "face_mask_path", "face_eye_mask_path", "tex_path"]:
            model_cfg[k] = os.path.join(flame_assets, os.path.basename(model_cfg[k]))

        # Use just the lower half of the face mask
        model_cfg["face_eye_mask_path"] = model_cfg["face_eye_mask_path"].replace("uv_face_eye_mask", "uv_dub_mask")
        model_cfg["face_mask_path"] = model_cfg["face_mask_path"].replace("uv_face_mask", "uv_dub_mask")

        # self.device = config.device
        self.shape_model = FLAME(model_cfg)
        self.mesh = trimesh.load(model_cfg.topology_path, process=False, maintain_order=True)
        self.faces = torch.from_numpy(self.mesh.faces)[None]
        self.renderer = ComaMeshRenderer('smooth', 'cuda')

    def forward(self, exp, jaw, shape=None):

        pose = torch.cat([torch.zeros((jaw.shape[0], 3), device=exp.device), jaw], dim=-1)
        verts, *_ = self.shape_model(expression_params=exp, pose_params=pose, shape_params=shape)
        images = self.renderer.render((verts, self.faces.to(exp.device).repeat(verts.shape[0], 1, 1)))
        return images

    def vertex_forward(self, exp, jaw, shape=None):
        pose = torch.cat([torch.zeros((jaw.shape[0], 3), device=exp.device), jaw], dim=-1)
        verts, *_ = self.shape_model(expression_params=exp, pose_params=pose, shape_params=shape)
        return verts

class TripleSyncnet(nn.Module):
    """SyncNet with 3 inputs: audio, video, and parameters.
   """
    def __init__(self, config, n_params=53, T=5):

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

        self.render_encoder = nn.ModuleList([
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
        self.lmk_idxs = [207, 427, 2]

        self.renderer = ParamsToImage(config)
        self.render_crops = {}
        self.video_crops = {}

    def get_mouth_bb(self, frame):

        frame_np = (frame.permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
        lmks = self.det.process(frame_np)
        if lmks.multi_face_landmarks is None:
            return -1

        lmks = lmks.multi_face_landmarks[0]
        lmks = lmks.landmark
        lmks = [[int(lmk.x * frame.shape[2]), int(lmk.y * frame.shape[1])] for lmk in lmks]
        lmks = [lmks[idx] for idx in self.lmk_idxs]
        x0 = min([lmk[0] for lmk in lmks])
        x1 = max([lmk[0] for lmk in lmks])
        y0 = min([lmk[1] for lmk in lmks])
        y1 = y0 + (x1 - x0)

        bb = [x0, y0, x1, y1]
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

    def audio_forward(self, audio, name='a'):
        audio_enc = None
        if audio is not None:
            audio_enc = self.audio_enc(audio[:, None, :, :])
            audio_enc = audio_enc.reshape((audio_enc.shape[0], -1))
            audio_enc = F.relu(audio_enc)

            # audio_enc = torch.sigmoid(audio_enc)
            audio_enc = F.normalize(audio_enc, p=2, dim=-1)
        return audio_enc

    def param_forward(self, params, name='a'):
        param_enc = None
        if params is not None:
            B, T, *_ = params.shape
            params = params.reshape((-1, params.shape[-1]))

            exp, pose = params[:, :50], params[:, 50:]
            render = self.renderer(exp, pose)[..., :3]

            render_crop = self.crop_video_to_mouth(render.permute((0, 3, 1, 2)).reshape((B, T, 3, 512, 512)))
            self.render_crops[name] = render_crop

            x = render_crop.reshape((B, T*3, *render_crop.shape[-2:]))

            for layer in self.render_encoder:
                x = layer(x)

            x = x.reshape((x.shape[0], -1))

            #param_enc = self.param_enc(params)
            param_enc = F.relu(x)
            #param_enc = torch.sigmoid(x)
            param_enc = F.normalize(param_enc, p=2, dim=-1)
        return param_enc

    def video_forward(self, video, name='a'):
        video_enc = None
        if video is not None:

            video = self.crop_video_to_mouth(video)
            self.video_crops[name] = video

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

            video = F.relu(video)
            #video = torch.sigmoid(video)
            video_enc = F.normalize(video, p=2, dim=-1)
        return video_enc


    # @profile
    def forward(self, audio=None, params=None, video=None, name='a'):
        # Audio = (B, 80, 16), params = (B, T, C), video = (B, T, 3, 256, 256)

        audio_enc = self.audio_forward(audio, name=name)
        param_enc = self.param_forward(params, name=name)
        video_enc = self.video_forward(video, name=name)

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

        #vp_loss_same = self._compute_loss(video_enc_a, param_enc_a, is_same=True)
        #vp_loss_different = self._compute_loss(video_enc_a, param_enc_b, is_same=False)

        av_loss_same = self._compute_loss(audio_enc_a, video_enc_a, is_same=True)
        av_loss_different = self._compute_loss(audio_enc_a, video_enc_b, is_same=False)

        #vp_loss = (vp_loss_same + vp_loss_different) / 2
        av_loss = (av_loss_same + av_loss_different) / 2

        #loss = (vp_loss + av_loss) / 2
        loss = av_loss

        return loss
