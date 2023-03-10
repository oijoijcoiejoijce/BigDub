import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader

from networks import UnetAdaIN
from networks.UNET import UnetRenderer
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


class Audio2Expression(pl.LightningModule):

    def __init__(self, config, IDs, nc=8):
        super().__init__()
        # self.save_hyperparameters()
        self.automatic_optimization = False

        self.nc = nc

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
        self.unet = UnetRenderer('UNET_5_level_ADAIN', nc, 3, norm_layer=nn.InstanceNorm2d)

        self.resize = Resize(config.getint("Model", "image_size"))

    def prepare_textures(self, IDs, tex_size=256, n_channels=8):
        textures = {}
        for ID in IDs:
            textures[ID] = torch.randn((1, n_channels, tex_size, tex_size), device=self.device, requires_grad=True)
        self.textures = nn.ParameterDict(textures)

    def training_step(self, batch):

        opt_tex, opt_img = self.optimizers()

        opt_tex.zero_grad()
        opt_img.zero_grad()

        # Get data
        params, frames, uv, inner_mask, outer_mask = self.prepare_input(batch)
        IDs = batch['ID']

        B, T, C, H, W = frames.shape

        # Prepare textures
        textures = torch.cat([self.textures[ID] for ID in IDs], dim=0)[:, None].repeat(1, T, 1, 1, 1)

        # Sample texture
        raster = F.grid_sample(textures.reshape((B*T, *textures.shape[2:])), uv.reshape((B*T, *uv.shape[2:])),
                               align_corners=False, padding_mode='zeros')
        raster = raster.reshape((B, T, *raster.shape[1:]))

        frames_pad = torch.cat((frames, torch.zeros(B, T, self.nc - 3, H, W, device=self.device)), dim=2)

        # Mask texture
        network_input = raster * inner_mask + (frames_pad * (1 - outer_mask))
        #network_input = frames

        # TODO: Condition on audio
        cond = torch.ones((B*T, 512), device=self.device)

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
        # loss = loss_img
        self.manual_backward(loss)

        self.log('loss', loss)
        self.log('loss_tex', loss_tex)
        self.log('loss_img', loss_img)
        # self.log_images('input', frames, self.current_epoch)

        opt_tex.step()
        opt_img.step()
        return loss

    def training_step(self, batch):

        # Get data
        params, frames, uv, inner_mask, outer_mask = self.prepare_input(batch)
        IDs = batch['ID']

        B, T, C, H, W = frames.shape

        # Prepare textures
        textures = torch.cat([self.textures[ID] for ID in IDs], dim=0)[:, None].repeat(1, T, 1, 1, 1)

        # Sample texture
        raster = F.grid_sample(textures.reshape((B*T, *textures.shape[2:])), uv.reshape((B*T, *uv.shape[2:])),
                               align_corners=False, padding_mode='zeros')
        raster = raster.reshape((B, T, *raster.shape[1:]))

        frames_pad = torch.cat((frames, torch.zeros(B, T, self.nc - 3, H, W, device=self.device)), dim=2)

        # Mask texture
        network_input = raster * inner_mask + (frames_pad * (1 - outer_mask))
        #network_input = frames

        # TODO: Condition on audio
        cond = torch.ones((B*T, 512), device=self.device)

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

        self.log('val/loss', loss, on_step=False, on_epoch=True)
        self.log('val/loss_tex', loss_tex, on_step=False, on_epoch=True)
        self.log('val/loss_img', loss_img, on_step=False, on_epoch=True)
        return loss

    def on_epoch_end(self) -> None:

        for i in range(1):
            # A bit hacky but it works
            gen, length = self.trainer._data_connector._val_dataloader_source.dataloader().dataset.get_video_generator()
            frames = []
            for frame_idx in range(length):
                batch = gen(frame_idx)

                for key in batch:
                    batch[key] = batch[key][None].to(self.device)

                params, frame, uv, inner_mask, outer_mask = self.prepare_input(batch)
                IDs = batch['ID']
                # Prepare textures

                B, T, C, H, W = frames.shape
                textures = torch.cat([self.textures[ID] for ID in IDs], dim=0)[:, None].repeat(1, T, 1, 1, 1)

                # Sample texture
                raster = F.grid_sample(textures.reshape((B * T, *textures.shape[2:])),
                                       uv.reshape((B * T, *uv.shape[2:])),
                                       align_corners=False, padding_mode='zeros')
                raster = raster.reshape((B, T, *raster.shape[1:]))

                frames_pad = torch.cat((frames, torch.zeros(B, T, self.nc - 3, H, W, device=self.device)), dim=2)

                # Mask texture
                network_input = raster * inner_mask + (frames_pad * (1 - outer_mask))
                # network_input = frames

                # TODO: Condition on audio
                cond = torch.ones((B * T, 512), device=self.device)

                # Resize
                network_input = self.resize(network_input.reshape((B * T, *network_input.shape[2:])))
                frames = self.resize(frames.reshape((B * T, *frames.shape[2:])))
                frames = frames.reshape((B, T, *frames.shape[1:]))

                # Train network
                network_output = self.unet(network_input, cond)

                frames.append(network_output[0, network_output.shape[1]//2].permute((1, 2, 0)).cpu().detach().numpy())

            video = np.stack(frames, axis=0)
            self.log('video', wandb.Video(video, fps=30, format="gif"))

    def prepare_input(self, batch):

        params = batch['params']
        frames = batch['frames']

        B, T, _ = params['shapecode'].shape
        params['images'] = torch.zeros((B, T, 3, 224, 224), device=self.device)

        params = {p: params[p].reshape(B*T, *params[p].shape[2:]) for p in params}
        frames = frames.reshape(B*T, *frames.shape[2:])

        out = self.emoca.decode_uv_mask_and_detail(params)

        # Warp uv and masks
        uv = out['predicted_images']
        inner_mask = out['inner_mask']
        outer_mask = out['outer_mask']

        uv = warp_image_tensor(uv, out['tform'], frames.shape[-1])
        inner_mask = warp_image_tensor(inner_mask, out['tform'], frames.shape[-1])
        outer_mask = warp_image_tensor(outer_mask, out['tform'], frames.shape[-1])

        uv = uv.permute((0, 2, 3, 1))[..., :2]

        frames = frames.reshape(B, T, *frames.shape[1:])
        inner_mask = inner_mask.reshape(B, T, *inner_mask.shape[1:])
        outer_mask = outer_mask.reshape(B, T, *outer_mask.shape[1:])
        uv = uv.reshape(B, T, *uv.shape[1:])

        return params, frames, uv, inner_mask, outer_mask

    def configure_optimizers(self):
        tex_opt = torch.optim.Adam(self.textures.values(), lr=1e-3)
        img_opt = torch.optim.Adam(self.unet.parameters(), lr=1e-4)
        return tex_opt, img_opt

def main():
    from Datasets import DubbingDataset, DataTypes
    from pytorch_lightning.callbacks import ModelSummary
    import configparser

    config_path = 'configs/Laptop.ini'
    config = configparser.ConfigParser()
    config.read(config_path)

    #index_path = os.path.join(config.get('Paths', 'data'), 'index.csv')
    data_root = config.get('Paths', 'data')

    torch.backends.cudnn.benchmark = True

    train_dataloader = torch.utils.data.DataLoader(
        DubbingDataset(data_root,
            data_types=[DataTypes.MEL, DataTypes.Params, DataTypes.Frames, DataTypes.ID], T=5),
        batch_size=1,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    val_dataloader = torch.utils.data.DataLoader(
        DubbingDataset(data_root,
            data_types=[DataTypes.MEL, DataTypes.Params, DataTypes.Frames, DataTypes.ID], split='test', T=5),
        batch_size=1,
        shuffle=True,
        num_workers=0
    )

    model = Audio2Expression(config, train_dataloader.dataset.ids)
    wandb_logger = WandbLogger(project='DubbingForExtras_NR')
    trainer = pl.Trainer(gpus=1, max_epochs=1,
                         callbacks=[ModelSummary(max_depth=2)], logger=wandb_logger,
                         default_root_dir="C:/Users/jacks/Documents/Data/DubbingForExtras/checkpoints/render/basic")
    trainer.fit(model, train_dataloader, val_dataloader)

if __name__ == '__main__':
    main()
