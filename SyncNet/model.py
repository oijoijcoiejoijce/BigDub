import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer

from networks.syncnet import TripleSyncnet
from pytorch_lightning.loggers import WandbLogger

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

class SyncNet(pl.LightningModule):

        def __init__(self, config):
            super().__init__()
            self.net = TripleSyncnet(config, 53)

        def prepare_parameters(self, params):
            exp, jaw = params['expcode'], params['posecode'][..., 3:]
            return torch.cat([exp, jaw], dim=-1)

        def training_step(self, batch, batch_idx):

            audio, params, frames = batch['MEL'], batch['params'], batch['frames']
            other_audio, other_params, other_frames = batch['other_MEL'], batch['other_params'], batch['other_frames']

            params = self.prepare_parameters(params)
            other_params = self.prepare_parameters(other_params)

            #mel_enc, params_enc, vid_enc = self.net(audio, params, frames, name='a')
            #other_mel_enc, other_params_enc, other_vid_enc = self.net(other_audio, other_params, other_frames, name='b')

            mel_enc, vid_enc = self.net(audio=audio, frames=frames, name='a')
            other_mel_enc, other_vid_enc = self.net(audio=other_audio, frames=other_frames, name='b')

            #loss = self.net.compute_loss(mel_enc, params_enc, vid_enc, other_mel_enc, other_params_enc, other_vid_enc)

            #loss = self.net.compute_loss(mel_enc, params_enc, vid_enc, other_mel_enc, other_params_enc, other_vid_enc)
            loss = self.net.compute_loss(audio_enc_a=mel_enc, video_enc_a=vid_enc,
                                         audio_enc_b=other_mel_enc, video_enc_b=other_vid_enc)

            self.log('train_loss', loss)
            return loss

        def validation_step(self, batch, batch_idx):

            audio, params, frames = batch['MEL'], batch['params'], batch['frames']
            other_audio, other_params, other_frames = batch['other_MEL'], batch['other_params'], batch['other_frames']

            params = self.prepare_parameters(params)
            other_params = self.prepare_parameters(other_params)

            mel_enc, params_enc, vid_enc = self.net(audio, params, frames, name='a')
            other_mel_enc, other_params_enc, other_vid_enc = self.net(other_audio, other_params, other_frames, name='b')
            loss = self.net.compute_loss(mel_enc, params_enc, vid_enc, other_mel_enc, other_params_enc, other_vid_enc)
            self.log('val_loss', loss, on_epoch=True, prog_bar=True)

        def configure_optimizers(self):
            return torch.optim.Adam(self.net.parameters(), lr=1e-3)

def main():
    from Datasets import DubbingDataset, DataTypes
    from pytorch_lightning.callbacks import ModelSummary
    import configparser
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/Laptop.ini')
    args = parser.parse_args()

    config_path = args.config
    config = configparser.ConfigParser()
    config.read(config_path)

    #index_path = os.path.join(config.get('Paths', 'data'), 'index.csv')
    data_root = config.get('Paths', 'data')
    batch_size = int(config.getint('SyncNet Training', 'batch_size'))
    grad_acc = int(config.getint('SyncNet Training', 'gradient_accumulate_every'))

    torch.backends.cudnn.benchmark = True

    train_dataloader = torch.utils.data.DataLoader(
        DubbingDataset(data_root,
            data_types=[DataTypes.MEL, DataTypes.Params, DataTypes.Frames], T=5, syncet=True),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    val_dataloader = torch.utils.data.DataLoader(
        DubbingDataset(data_root,
            data_types=[DataTypes.MEL, DataTypes.Params, DataTypes.Frames], split='test', T=5,
                       syncet=True),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    wandb_logger = WandbLogger(project='DubbingForExtras_Syncnet')
    model = SyncNet(config)
    trainer = pl.Trainer(gpus=1, max_epochs=400,
                         callbacks=[ModelSummary(max_depth=2)],
                         default_root_dir="C:/Users/jacks/Documents/Data/DubbingForExtras/checkpoints/sync/basic",
                         logger=wandb_logger, gradient_clip_val=1.0, accumulate_grad_batches=grad_acc)
    trainer.fit(model, train_dataloader, val_dataloader)


if __name__ == '__main__':
    main()
