import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import sys
from pathlib import Path

import cv2

sys.path.append(str(Path(__file__).parent.parent))

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from networks import UnetAdaIN
from networks.UNET import UnetRenderer
from networks.networks import define_D
from networks.VGG import VGGLOSS
from networks import Discriminator, AudioEncoder
from EMOCA_lite.model import DecaModule
from pytorch_lightning.loggers import WandbLogger
from skimage.transform import warp
import wandb

from torchvision.transforms import Compose, ToTensor, Resize

import os
from omegaconf import OmegaConf
import numpy as np

def warp_image_tensor(tensor, tform, img_size):
    *expand_dims, C, H, W = tensor.shape
    tensor = tensor.reshape(-1, C, H, W)
    tensor = tensor.permute(0, 2, 3, 1)
    tensor = tensor.cpu().numpy()
    out = []
    for i in range(tensor.shape[0]):
        out.append(warp(tensor[i], tform[i].detach().cpu().numpy(), output_shape=(img_size, img_size)))
    tensor = np.stack(out)
    tensor = torch.from_numpy(tensor).permute(0, 3, 1, 2).to(0)
    return tensor.reshape((*expand_dims, C, img_size, img_size))


class NeuralRenderer(pl.LightningModule):

    def __init__(self, config, IDs, nc=16, T=5, logger=None):
        super().__init__()
        # self.save_hyperparameters()
        self.automatic_optimization = False

        self.config = config
        self.use_audio = config.getboolean("Model", "use_audio")
        self.wandb_logger = logger
        self.nc = nc
        self.T = T

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

        checkpoint_kwargs = {
            "model_params": model_cfg,
            "stage_name": "testing",
        }

        self.emoca = DecaModule.load_from_checkpoint(emoca_checkpoint, strict=False, **checkpoint_kwargs).to(self.device)
        self.emoca.eval()

        self.prepare_textures(IDs, n_channels=nc)
        # self.unet = UnetAdaIN(3, 3).to(self.device)
        self.unet = UnetRenderer('UNET_3_level_DC', nc + 3, 3, norm_layer=nn.InstanceNorm2d)

        # if self.use_audio:
        self.audio_enc = AudioEncoder()

        self.discriminator = define_D(3*T, 128, 'basic', n_layers_D=8, norm='instance', init_type='normal')
        self.vgg = VGGLOSS().to(self.device)

        self.resize = Resize(config.getint("Model", "image_size"))

    def prepare_textures(self, IDs, tex_size=256, n_channels=8):
        textures = {}
        for ID in IDs:
            textures[ID] = nn.Parameter(torch.randn((1, n_channels, tex_size, tex_size),
                                                    device=self.device, requires_grad=True))
        self.textures = nn.ParameterDict(textures)

    def prepare_network_input(self, frames, uv, inner_mask, outer_mask, IDs, other_frames):
        B, T, C, H, W = frames.shape

        # Prepare textures
        textures = torch.cat([self.textures[ID] for ID in IDs], dim=0)[:, None].repeat(1, T, 1, 1, 1)

        # Sample texture
        raster = F.grid_sample(textures.reshape((B * T, *textures.shape[2:])), uv.reshape((B * T, *uv.shape[2:])),
                               align_corners=False, padding_mode='zeros')
        raster = raster.reshape((B, T, *raster.shape[1:]))

        frames_pad = torch.cat((frames, torch.zeros(B, T, self.nc - 3, H, W, device=self.device)), dim=2)

        # Mask texture
        network_input = raster * inner_mask + (frames_pad * (1 - outer_mask))
        network_input = torch.cat((network_input, other_frames), dim=2)
        return network_input

    def train_discriminator(self, frames, generator_output, opt_discriminator):

        B, T, C, H, W = generator_output.shape

        network_output = generator_output.reshape((B, T*C, H, W))
        frames = frames.reshape((B, T*C, H, W))

        # Train discriminator
        D_pred_real = self.discriminator(frames)
        D_pred_fake = self.discriminator(network_output.detach())

        D_loss_real = F.mse_loss(D_pred_real, torch.ones_like(D_pred_real))
        D_loss_fake = F.mse_loss(D_pred_fake, torch.zeros_like(D_pred_fake))

        D_loss = ((D_loss_real + D_loss_fake)/2)

        loss = D_loss * 0.02

        wandb.log({"D_loss": D_loss, "D_loss_real": D_loss_real, "D_loss_fake": D_loss_fake},
                  step=self.trainer.global_step)
        opt_discriminator.zero_grad()

        if D_loss > 0.01:
            self.manual_backward(loss)

        opt_discriminator.step()

    def forward(self, batch):
        # Get data
        params, frames, uv, inner_mask, outer_mask, mouth_mask, other_frames = self.prepare_input(batch)
        IDs = batch['ID']

        B, T, C, H, W = frames.shape

        # Prepare network input
        network_input = self.prepare_network_input(frames, uv, inner_mask, outer_mask, IDs, other_frames)

        # TODO: Condition on audio
        if self.use_audio:
            audio = batch['ind_MEL']
            audio = audio[:, :,  None, :, :].reshape((B*T, 1, *audio.shape[2:]))
            cond = self.audio_enc(audio).reshape((B * T, -1))
        else:
            cond = torch.ones((B * T, 512), device=self.device)

        network_input = self.resize(network_input.reshape((B * T, *network_input.shape[2:])))
        frames = self.resize(frames.reshape((B * T, *frames.shape[2:])))
        frames = frames.reshape((B, T, *frames.shape[1:]))

        # Train network
        network_output = self.unet(network_input, cond)

        network_output = network_output.reshape((B, T, *network_output.shape[1:]))
        network_input = network_input.reshape((B, T, *network_input.shape[1:]))

        return network_output, frames, network_input
    def train_generator(self, network_output, frames, network_input, opt_tex, opt_img, opt_audio):

        B, T, C, H, W = frames.shape

        loss_tex = (network_input[:, :, :3] - frames).abs().mean()
        loss_img = (network_output - frames).abs().mean()
        loss_vgg = self.vgg(network_output.reshape((B*T, C, W, H)), frames.reshape((B*T, C, W, H)))

        network_output = network_output.reshape((B, T*C, H, W))
        D_pred_fake = self.discriminator(network_output)
        loss_G_adv = F.mse_loss(D_pred_fake, torch.ones_like(D_pred_fake))

        loss = (1.0 * loss_img) + (1.0 * loss_tex) + (0.02 * loss_G_adv) + (1.0 * loss_vgg)

        # loss = loss_img
        opt_tex.zero_grad()
        opt_img.zero_grad()
        opt_audio.zero_grad()

        self.manual_backward(loss)

        # Log images
        wandb.log({"loss": loss,
                   "loss_tex": loss_tex,
                   "loss_img": loss_img,
                   "loss_G_adv": loss_G_adv,
                   "loss_vgg": loss_vgg,
                   }, step=self.trainer.global_step)

        opt_tex.step()
        opt_img.step()
        opt_audio.step()
        return (loss_img + loss_tex + loss_vgg)  # Do not include adversarial loss for metrics


    def training_step(self, batch):

        opt_tex, opt_img, opt_discriminator, opt_audio = self.optimizers()
        scheduler_tex, scheduler_img, scheduler_discriminator, scheduler_audio = self.lr_schedulers()

        network_output, frames, network_input = self.forward(batch)

        self.train_discriminator(frames, network_output, opt_discriminator)

        loss = self.train_generator(network_output, frames, network_input, opt_tex, opt_img, opt_audio)
        if self.current_epoch > 100:
            scheduler_tex.step()
            scheduler_img.step()
            scheduler_discriminator.step()
            scheduler_audio.step()

        return loss

    def on_validation_epoch_start(self) -> None:
        self.losses = []
        self.losses_tex = []
        self.losses_img = []

    def on_validation_epoch_end(self) -> None:
        wandb.log({"val_loss": np.mean(self.losses),
                   "val_loss_tex": np.mean(self.losses_tex),
                   "val_loss_img": np.mean(self.losses_img)}, step=self.trainer.global_step)

    def validation_step(self, batch, *args):

        # Get data
        params, frames, uv, inner_mask, outer_mask, mouth_mask, other_frames = self.prepare_input(batch)
        IDs = batch['ID']

        B, T, C, H, W = frames.shape

        # Prepare network input
        network_input = self.prepare_network_input(frames, uv, inner_mask, outer_mask, IDs, other_frames)

        # TODO: Condition on audio
        if self.use_audio:
            audio = batch['ind_MEL']
            audio = audio[:, :, None, :, :].reshape((B * T, 1, *audio.shape[2:]))
            cond = self.audio_enc(audio).reshape((B * T, -1))
        else:
            cond = torch.ones((B * T, 512), device=self.device)

        # Resize
        network_input = self.resize(network_input.reshape((B*T, *network_input.shape[2:])))
        frames = self.resize(frames.reshape((B*T, *frames.shape[2:])))
        frames = frames.reshape((B, T, *frames.shape[1:]))

        # Train network
        network_output = self.unet(network_input, cond)
        network_output = network_output.reshape((B, T, *network_output.shape[1:]))
        network_input = network_input.reshape((B, T, *network_input.shape[1:]))

        loss_tex = (network_input[:, :, :3] - frames).abs().mean()
        loss_img = (network_output - frames).abs().mean()
        loss = loss_tex + loss_img

        self.losses.append(loss.item())
        self.losses_tex.append(loss_tex.item())
        self.losses_img.append(loss_img.item())

        return loss

    def dict_to_torch(self, d, expand_batch=False):
        for key in d:
            if isinstance(d[key], dict):
                d[key] = self.dict_to_torch(d[key], expand_batch=expand_batch)
            elif isinstance(d[key], str):
                if expand_batch:
                    d[key] = [d[key]]
            else:
                d[key] = torch.tensor(d[key], device=self.device)
                if expand_batch:
                    d[key] = d[key][None]
        return d

    def on_train_epoch_end(self) -> None:

        if self.wandb_logger is None:
            return

        if self.current_epoch % 5 != 0:
            return

        for i in range(3):
            # A bit hacky but it works
            gen, length = self.trainer._data_connector._val_dataloader_source.dataloader().dataset.get_video_generator()
            video = self.create_video_from_generator(gen, length)
            wandb.log({f'video_{i}': wandb.Video(video, fps=30, format="mp4")})

    def create_video_from_generator(self, gen, length):
        video = []
        audios = []
        with torch.no_grad():
            for frame_idx in range(length):
                batch = gen(frame_idx)

                batch = self.dict_to_torch(batch, expand_batch=True)

                params, frames, uv, inner_mask, outer_mask, mouth_mask, other_frames = self.prepare_input(batch)
                IDs = batch['ID']
                # Prepare textures

                B, T, C, H, W = frames.shape

                # Prepare network input
                network_input = self.prepare_network_input(frames, uv, inner_mask, outer_mask, IDs, other_frames)

                # TODO: Condition on audio
                if self.use_audio:
                    audio = batch['ind_MEL']
                    audio = audio[:, :, None, :, :].reshape((B * T, 1, *audio.shape[2:]))
                    cond = self.audio_enc(audio).reshape((B * T, -1))
                    audios.append(cond)
                else:
                    cond = torch.ones((B * T, 512), device=self.device)
                    audios.append(cond)

                # Resize
                network_input = self.resize(network_input.reshape((B * T, *network_input.shape[2:])))
                frames = self.resize(frames.reshape((B * T, *frames.shape[2:])))
                frames = frames.reshape((B, T, *frames.shape[1:]))

                # Train network
                network_output = self.unet(network_input, cond)
                network_output = network_output.reshape((B, T, *network_output.shape[1:]))

                #vid_frame = torch.cat([frames[0, network_output.shape[1] // 2],
                #                       network_output[0, network_output.shape[1] // 2]],
                #                      dim=2).cpu().detach().numpy()

                np_frame = (frames[0, network_output.shape[1] // 2].permute((1, 2, 0)).cpu().detach().numpy() * 255).astype(np.uint8)
                np_output = (network_output[0, network_output.shape[1] // 2].permute((1, 2, 0)).cpu().detach().numpy().clip(0, 1) * 255).astype(np.uint8)
                np_outer_mask = (outer_mask[0, network_output.shape[1] // 2].permute((1, 2, 0)).cpu().detach().numpy() * 255).astype(np.uint8)
                np_outer_mask = cv2.resize(np_outer_mask, (np_output.shape[1], np_output.shape[0]))

                np_inner_mask = (inner_mask[0, network_output.shape[1] // 2].permute((1, 2, 0)).cpu().detach().numpy() * 255).astype(np.uint8)
                np_inner_mask = cv2.resize(np_inner_mask, (np_output.shape[1], np_output.shape[0]))

                np_mouth_mask = (mouth_mask[0, network_output.shape[1] // 2].permute(
                    (1, 2, 0)).cpu().detach().numpy() * 255).astype(np.uint8)
                np_mouth_mask = cv2.resize(np_mouth_mask, (np_output.shape[1], np_output.shape[0]))

                np_difference_mask = np.logical_xor(np_mouth_mask==0, np_outer_mask==0).astype(np.uint8) * 255
                np_clone = np_frame.copy()
                np_clone[np_mouth_mask > 0] = np_output[np_mouth_mask > 0]

                # blended = cv2.seamlessClone(np_output, np_clone, np_difference_mask, (np_output.shape[1] // 2, np_output.shape[0] // 2), cv2.MIXED_CLONE)

                vid_frame = np.concatenate([np_frame, np_clone], axis=1).transpose((2, 0, 1))

                video.append(vid_frame)

            video = np.stack(video, axis=0) #.clip(0, 1) * 255).astype(np.uint8)
            print(torch.cat(audios, dim=0).var(dim=0).mean().item())
        return video

    def prepare_input(self, batch):

        params = batch['params']
        frames = batch['frames']
        other_frames = batch['other_frames']

        B, T, _ = params['shapecode'].shape
        params['images'] = torch.zeros((B, T, 3, 224, 224), device=self.device)

        params = {p: params[p].reshape(B*T, *params[p].shape[2:]) for p in params}
        frames = frames.reshape(B*T, *frames.shape[2:])

        with torch.no_grad():
            out = self.emoca.decode_uv_mask_and_detail(params)

        # Warp uv and masks
        uv = out['predicted_images']
        inner_mask = out['inner_mask']
        outer_mask = out['outer_mask']
        inner_and_mouth_mask = out['inner_and_mouth_mask']

        uv = warp_image_tensor(uv, out['tform'], frames.shape[-1])
        inner_mask = warp_image_tensor(inner_mask, out['tform'], frames.shape[-1])
        outer_mask = warp_image_tensor(outer_mask, out['tform'], frames.shape[-1])
        inner_and_mouth_mask = warp_image_tensor(inner_and_mouth_mask, out['tform'], frames.shape[-1])

        uv = uv.permute((0, 2, 3, 1))[..., :2]

        frames = frames.reshape(B, T, *frames.shape[1:])
        inner_mask = inner_mask.reshape(B, T, *inner_mask.shape[1:])
        outer_mask = outer_mask.reshape(B, T, *outer_mask.shape[1:])
        inner_and_mouth_mask = inner_and_mouth_mask.reshape(B, T, *inner_and_mouth_mask.shape[1:])
        uv = uv.reshape(B, T, *uv.shape[1:])

        return params, frames, uv, inner_mask, outer_mask, inner_and_mouth_mask, other_frames

    def configure_optimizers(self):
        tex_opt = torch.optim.Adam(self.textures.values(), lr=2e-3, betas=(0.5, 0.999))
        img_opt = torch.optim.Adam(self.unet.parameters(), lr=2e-4, betas=(0.5, 0.999))
        dis_opt = torch.optim.Adam(self.discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))

        audio_opt = torch.optim.Adam(self.audio_enc.parameters(), lr=2e-4, betas=(0.5, 0.999))

        tex_scheduler = torch.optim.lr_scheduler.LinearLR(tex_opt, 1.0, 0.0, 100)
        img_scheduler = torch.optim.lr_scheduler.LinearLR(img_opt, 1.0, 0.0, 100)
        dis_scheduler = torch.optim.lr_scheduler.LinearLR(dis_opt, 1.0, 0.0, 100)
        audio_scheduler = torch.optim.lr_scheduler.LinearLR(audio_opt, 1.0, 0.0, 100)
        return [tex_opt, img_opt, dis_opt, audio_opt], [tex_scheduler, img_scheduler, dis_scheduler, audio_scheduler]

    def finetune(self, dataloader, ID_name, n_epoch=1):
        if ID_name in self.textures:
            texture = self.textures[ID_name]
        else:
            return

        # Optimizer takes texture only
        optim = torch.optim.Adam(self.unet.parameters(), lr=1e-5)

        wandb.init(project="DubbingForExtras_NR_ft")
        iters = 0

        for epoch in range(n_epoch):
            for batch in tqdm(dataloader):

                # Prepare input
                batch = self.dict_to_torch(batch, expand_batch=True)
                params, frames, uv, inner_mask, outer_mask, mouth_mask, other_frames = self.prepare_input(batch)
                IDs = batch['ID']

                # Prepare network input
                B, T, C, H, W = frames.shape

                # Prepare textures
                textures = torch.cat([texture for ID in IDs], dim=0)[:, None].repeat(1, T, 1, 1, 1)

                # Sample texture
                raster = F.grid_sample(textures.reshape((B * T, *textures.shape[2:])),
                                       uv.reshape((B * T, *uv.shape[2:])),
                                       align_corners=False, padding_mode='zeros')
                raster = raster.reshape((B, T, *raster.shape[1:]))

                frames_pad = torch.cat((frames, torch.zeros(B, T, self.nc - 3, H, W, device=self.device)), dim=2)

                # Mask texture
                network_input = raster * inner_mask + (frames_pad * (1 - outer_mask))
                network_input = torch.cat((network_input, other_frames), dim=2)

                network_input = self.resize(network_input.reshape((B * T, *network_input.shape[2:])))
                frames = self.resize(frames.reshape((B * T, *frames.shape[2:])))
                frames = frames.reshape((B, T, *frames.shape[1:]))

                if self.use_audio:
                    audio = batch['ind_MEL']
                    audio = audio[:, :, None, :, :].reshape((B * T, 1, *audio.shape[2:]))
                    cond = self.audio_enc(audio).reshape((B * T, -1))
                else:
                    cond = torch.ones((B * T, 512), device=self.device)

                # Train network
                network_output = self.unet(network_input, cond)
                network_output = network_output.reshape((B, T, *network_output.shape[1:]))
                network_input = network_input.reshape((B, T, *network_input.shape[1:]))

                loss_tex = (network_input[:, :, :3] - frames).abs().mean()
                loss_img = (network_output - frames).abs().mean()
                loss_vgg = self.vgg(network_output.reshape((B * T, *network_output.shape[2:])),
                                    frames.reshape((B * T, *frames.shape[2:])))
                loss = loss_tex + loss_img + loss_vgg
                # loss = loss_img

                optim.zero_grad()
                loss.backward()
                optim.step()

                iters += 1
                if iters > 1000:
                    break

                wandb.log({'finetune/loss': loss, 'finetune/loss_tex': loss_tex,
                           'finetune/loss_img': loss_img})


    def fit_to_new_ID(self, dataloader, n_epoch=1, ID_name=None):
        """ Fit the model to a new ID, optimizing only the texture
            This allows to train the model on a new ID without having to retrain the whole model
        """
        # Create a new neural texture
        texture = torch.randn_like(self.textures[list(self.textures.keys())[0]])

        # If the ID already exists, use the texture of the ID
        if ID_name is not None:
            if ID_name in self.textures:
                texture = self.textures[ID_name]

        # Optimizer takes texture only
        optim = torch.optim.Adam([texture], lr=1e-3)

        wandb.init(project="DubbingForExtras_NR_ft")
        iters = 0

        for epoch in range(n_epoch):
            for batch in tqdm(dataloader):

                # Prepare input
                batch = self.dict_to_torch(batch, expand_batch=True)
                params, frames, uv, inner_mask, outer_mask, mouth_mask, other_frames = self.prepare_input(batch)
                IDs = batch['ID']

                # Prepare network input
                B, T, C, H, W = frames.shape

                # Prepare textures
                textures = torch.cat([texture for ID in IDs], dim=0)[:, None].repeat(1, T, 1, 1, 1)

                # Sample texture
                raster = F.grid_sample(textures.reshape((B * T, *textures.shape[2:])),
                                       uv.reshape((B * T, *uv.shape[2:])),
                                       align_corners=False, padding_mode='zeros')
                raster = raster.reshape((B, T, *raster.shape[1:]))

                frames_pad = torch.cat((frames, torch.zeros(B, T, self.nc - 3, H, W, device=self.device)), dim=2)

                # Mask texture
                network_input = raster * inner_mask + (frames_pad * (1 - outer_mask))
                network_input = torch.cat((network_input, other_frames), dim=2)

                network_input = self.resize(network_input.reshape((B * T, *network_input.shape[2:])))
                frames = self.resize(frames.reshape((B * T, *frames.shape[2:])))
                frames = frames.reshape((B, T, *frames.shape[1:]))

                # TODO: Condition on audio
                if self.use_audio:
                    audio = batch['ind_MEL']
                    audio = audio[:, :, None, :, :].reshape((B * T, 1, *audio.shape[2:]))
                    cond = self.audio_enc(audio).reshape((B * T, -1))
                else:
                    cond = torch.ones((B * T, 512), device=self.device)

                # Train network
                network_output = self.unet(network_input, cond)
                network_output = network_output.reshape((B, T, *network_output.shape[1:]))
                network_input = network_input.reshape((B, T, *network_input.shape[1:]))

                loss_tex = (network_input[:, :, :3] - frames).abs().mean()
                loss_img = (network_output - frames).abs().mean()
                loss_vgg = self.vgg(network_output.reshape((B * T, *network_output.shape[2:])),
                                    frames.reshape((B * T, *frames.shape[2:])))
                loss = loss_tex + loss_img + loss_vgg
                # loss = loss_img

                optim.zero_grad()
                loss.backward()
                optim.step()

                iters += 1
                if iters > 1000:
                    break

                wandb.log({'finetune/loss': loss, 'finetune/loss_tex': loss_tex,
                           'finetune/loss_img': loss_img})

        self.textures[ID_name] = texture

def main():
    from Datasets import DubbingDataset, DataTypes
    from pytorch_lightning.callbacks import ModelSummary
    import configparser
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/Laptop.ini')
    parser.add_argument('--load_checkpoint', type=str, default='')
    parser.add_argument('--restrict_to_ID', type=str, default='')
    parser.add_argument('--max_vid_idx', type=int, default=-1)
    args = parser.parse_args()

    config_path = args.config
    config = configparser.ConfigParser()
    config.read(config_path)

    #index_path = os.path.join(config.get('Paths', 'data'), 'index.csv')
    data_root = config.get('Paths', 'data')
    batch_size = int(config.getint('NR Training', 'batch_size'))
    n_train_workers = int(config.getint('NR Training', 'n_train_workers'))
    n_val_workers = int(config.getint('NR Training', 'n_val_workers'))
    checkpoint_path = config.get('Paths', 'NR checkpoint')

    if args.restrict_to_ID != '':
        checkpoint_path = os.path.join(checkpoint_path, args.restrict_to_ID+str(args.max_vid_idx))
    else:
        checkpoint_path = os.path.join(checkpoint_path, 'generic')

    torch.backends.cudnn.benchmark = True


    if args.restrict_to_ID != '':
        restrict_videos = None if args.max_vid_idx == -1 else list(range(args.max_vid_idx))
        videos = [f'{args.restrict_to_ID}_{i}' for i in restrict_videos]
        train_dataloader = torch.utils.data.DataLoader(
            DubbingDataset(data_root,
                           data_types=[DataTypes.MEL, DataTypes.Params, DataTypes.Frames, DataTypes.ID],
                           T=5, syncet=True, split='all', restrict_videos=videos, fix_ID=args.restrict_to_ID),
            batch_size=batch_size,
            shuffle=True,
            num_workers=n_train_workers,
            pin_memory=True
        )

        val_videos = [f'{args.restrict_to_ID}_{i}' for i in [36, 37]]
        val_dataloader = torch.utils.data.DataLoader(
            DubbingDataset(data_root,
                           data_types=[DataTypes.MEL, DataTypes.Params, DataTypes.Frames, DataTypes.ID],
                           split='all', T=5, syncet=True, fix_ID=args.restrict_to_ID, restrict_videos=val_videos),
            batch_size=batch_size,
            shuffle=False,
            num_workers=n_val_workers
        )

    else:
        train_dataloader = torch.utils.data.DataLoader(
            DubbingDataset(data_root,
                data_types=[DataTypes.MEL, DataTypes.Params, DataTypes.Frames, DataTypes.ID],
                           T=5, syncet=True, split='train'),
            batch_size=batch_size,
            shuffle=True,
            num_workers=n_train_workers,
            pin_memory=True
        )

        val_dataloader = torch.utils.data.DataLoader(
            DubbingDataset(data_root,
                data_types=[DataTypes.MEL, DataTypes.Params, DataTypes.Frames, DataTypes.ID],
                           split='test', T=5, syncet=True),
            batch_size=batch_size,
            shuffle=False,
            num_workers=n_val_workers
        )
    wandb_logger = WandbLogger(project='DubbingForExtras_NR')
    model = NeuralRenderer(config, train_dataloader.dataset.ids, logger=wandb_logger)

    trainer = pl.Trainer(gpus=1, max_epochs=200,
                         callbacks=[ModelSummary(max_depth=2)],
                         default_root_dir=checkpoint_path, num_sanity_val_steps=0)

    if args.load_checkpoint:
        trainer.fit(model, train_dataloader, val_dataloader, ckpt_path=args.load_checkpoint)
    else:
        trainer.fit(model, train_dataloader, val_dataloader)


if __name__ == '__main__':
    main()
