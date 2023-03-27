import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb

from networks import AudioEncoder
from pytorch_lightning.loggers import WandbLogger
from networks.Wav2VecFeatures import Wav2Vec2Model
from networks.syncnet import ParamsToImage
import numpy as np


class Audio2ExpressionLSTM(nn.Module):

    def __init__(self, n_params, n_features=80, n_hidden=64, n_layers=1, T=5):
        super(Audio2ExpressionLSTM, self).__init__()

        self.n_params = n_params
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.T = T

        self.audio_feature_map = nn.Linear(768, n_hidden)

        self.lstm = nn.LSTM(n_hidden, n_hidden, n_layers, batch_first=True)
        self.fc = nn.Linear(n_hidden, n_params)

    def forward(self, audio, n_frames=None):
        # x = (B, T, 80, 16)
        #audio_enc = self.audio_encoder(audio, frame_num=n_frames).last_hidden_state

        audio_enc = self.audio_feature_map(audio)
        # audio_enc = audio_enc.reshape((B, T, -1))
        x, _ = self.lstm(audio_enc)
        return x


class Audio2Expression(pl.LightningModule):

    def __init__(self, config, logger):
        super().__init__()
        self.wandb_logger = logger
        self.save_hyperparameters()
        self.model = Audio2ExpressionLSTM(53)
        self.style_encoder = nn.LSTM(53, 64, 1, batch_first=True)
        self.fc = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 53))

        self.loss = nn.MSELoss()
        self.metric = None
        self.renderer = ParamsToImage(config)

    def prep_data(self, batch):
        audio = batch['wav2vec']
        params = batch['params']
        other_params = batch['other_params']

        target_params = torch.cat([params['expcode'], params['posecode'][..., 3:]], dim=-1)
        other_params = torch.cat([other_params['expcode'], other_params['posecode'][..., 3:]], dim=-1)

        return audio, target_params, other_params

    def forward(self, audio, other_params):
        _, (style, _) = self.style_encoder(other_params)
        style = style.reshape((other_params.shape[0], -1, 64))
        style = style.repeat(1, audio.shape[1], 1)

        pred_tokens = self.model(audio, n_frames=audio.shape[1])
        stylised = style + pred_tokens
        pred_params = self.fc(stylised)
        return pred_params

    def training_step(self, batch):

        audio, target_params, other_params = self.prep_data(batch)
        pred_params = self(audio, other_params)

        target_exp, target_jaw = target_params[..., :50], target_params[..., 50:]
        pred_exp, pred_jaw = pred_params[..., :50], pred_params[..., 50:]

        target_verts = self.renderer.vertex_forward(target_exp.reshape((-1, 50)), target_jaw.reshape((-1, 3)))
        pred_verts = self.renderer.vertex_forward(pred_exp.reshape((-1, 50)), pred_jaw.reshape((-1, 3)))

        #recon_loss = self.loss(pred_params, target_params)
        #vel_loss = self.loss(pred_params[:, :-1] - pred_params[:, 1:], target_params[:, :-1] - target_params[:, 1:])

        recon_loss = self.loss(pred_verts, target_verts)
        vel_loss = self.loss(pred_verts[:, :-1] - pred_verts[:, 1:], target_verts[:, :-1] - target_verts[:, 1:])

        loss = (recon_loss + (10 * vel_loss)) * 1e6
        wandb.log({'train_loss': loss, 'train_recon_loss': recon_loss, 'train_vel_loss': vel_loss},
                  step=self.trainer.global_step)
        return loss

    def on_validation_epoch_start(self):
        self.losses = []
        self.recon_losses = []
        self.vel_losses = []

    def on_validation_epoch_end(self):
        wandb.log({'val_loss': np.mean(self.losses), 'val_recon_loss': np.mean(self.recon_losses), 'val_vel_loss': np.mean(self.vel_losses)},
                  step=self.trainer.global_step)

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

    def create_video(self, batch):

        with torch.no_grad():
            batch = self.dict_to_torch(batch, expand_batch=True)
            audio, target_params, other_params = self.prep_data(batch)
            pred_params = self(audio, other_params)

            exp, pose = pred_params[0, ..., :50], pred_params[0, ..., 50:]
            exp_real, pose_real = target_params[0, ..., :50], target_params[0, ..., 50:]

            vid = []
            for i in range(0, exp.shape[0], 5):
                x = self.renderer(exp[i:i+5], pose[i:i+5])
                y = self.renderer(exp_real[i:i+5], pose_real[i:i+5])
                vid.append(torch.cat((x, y), dim=2))
            vid = torch.cat(vid, dim=0).clip(0, 1) * 255
            return vid.detach().cpu()[..., :3].permute((0, 3, 1, 2)).numpy()

    def on_train_epoch_end(self):

        if self.wandb_logger is None:
            return

        if self.current_epoch % 50 != 0:
            return

        for i in range(3):
            batch = self.trainer._data_connector._val_dataloader_source.dataloader().dataset.get_full_video()
            video = self.create_video(batch)
            self.wandb_logger.experiment.log({f'video_{i}': wandb.Video(video, fps=30, format="mp4")},
                                             step=self.trainer.global_step)

    def validation_step(self, batch, idx):
        audio, target_params, other_params = self.prep_data(batch)
        pred_params = self(audio, other_params)

        target_exp, target_jaw = target_params[..., :50], target_params[..., 50:]
        pred_exp, pred_jaw = pred_params[..., :50], pred_params[..., 50:]

        target_verts = self.renderer.vertex_forward(target_exp.reshape((-1, 50)), target_jaw.reshape((-1, 3)))
        pred_verts = self.renderer.vertex_forward(pred_exp.reshape((-1, 50)), pred_jaw.reshape((-1, 3)))

        # recon_loss = self.loss(pred_params, target_params)
        # vel_loss = self.loss(pred_params[:, :-1] - pred_params[:, 1:], target_params[:, :-1] - target_params[:, 1:])

        recon_loss = self.loss(pred_verts, target_verts)
        vel_loss = self.loss(pred_verts[:, :-1] - pred_verts[:, 1:], target_verts[:, :-1] - target_verts[:, 1:])

        loss = (recon_loss + (10 * vel_loss)) * 1e6

        self.losses.append(loss.item())
        self.recon_losses.append(recon_loss.item())
        self.vel_losses.append(vel_loss.item())

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

def main():
    from Datasets import DubbingDataset, DataTypes
    from pytorch_lightning.callbacks import ModelSummary

    import argparse
    import configparser

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/Laptop.ini')
    args = parser.parse_args()

    config_path = args.config
    config = configparser.ConfigParser()
    config.read(config_path)

    train_dataloader = torch.utils.data.DataLoader(
        DubbingDataset('C:/Users/jacks/Documents/Data/DubbingForExtras/v3',
            data_types=[DataTypes.WAV2VEC, DataTypes.Params], T=21),
        batch_size=8,
        shuffle=True,
        num_workers=0,
    )

    val_dataloader = torch.utils.data.DataLoader(
        DubbingDataset('C:/Users/jacks/Documents/Data/DubbingForExtras/v3',
            data_types=[DataTypes.WAV2VEC, DataTypes.Params], split='test', T=21),
        batch_size=8,
        shuffle=True,
        num_workers=0
    )

    logger = WandbLogger(project='Audio2Expression')
    model = Audio2Expression(config, logger)

    trainer = pl.Trainer(gpus=1, max_epochs=1000, gradient_clip_val=0.5,
                         callbacks=[ModelSummary(max_depth=-1)], logger=logger)
    trainer.fit(model, train_dataloader, val_dataloader)

if __name__ == '__main__':
    main()

