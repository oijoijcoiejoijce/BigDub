import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader

from networks import AudioEncoder
from pytorch_lightning.loggers import WandbLogger
from networks.Wav2VecFeatures import Wav2Vec2Model


class Audio2ExpressionLSTM(nn.Module):

    def __init__(self, n_params, n_features=80, n_hidden=64, n_layers=1, T=5):
        super(Audio2ExpressionLSTM, self).__init__()

        self.n_params = n_params
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.T = T

        self.audio_encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        # wav2vec 2.0 weights initialization
        self.audio_encoder.feature_extractor._freeze_parameters()
        self.audio_feature_map = nn.Linear(768, n_hidden)

        self.lstm = nn.LSTM(n_hidden, n_hidden, n_layers, batch_first=True)
        self.fc = nn.Linear(n_hidden, n_params)

    def forward(self, audio, n_frames=None):
        # x = (B, T, 80, 16)
        audio_enc = self.audio_encoder(audio, frame_num=n_frames).last_hidden_state
        audio_enc = self.audio_feature_map(audio_enc)
        # audio_enc = audio_enc.reshape((B, T, -1))
        x, _ = self.lstm(audio_enc)
        return x


class Audio2Expression(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
        self.model = Audio2ExpressionLSTM(53)
        self.style_encoder = nn.LSTM(53, 64, 1, batch_first=True)
        self.fc = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 53))

        self.loss = nn.MSELoss()
        self.metric = None

    def training_step(self, batch):
        audio = batch['audio']
        params = batch['params']
        other_params = batch['other_params']

        target_params = torch.cat([params['expcode'], params['posecode'][..., 3:]], dim=-1)
        other_params = torch.cat([other_params['expcode'], other_params['posecode'][..., 3:]], dim=-1)

        _, (style, _) = self.style_encoder(other_params)
        style = style.repeat(1, target_params.shape[1], 1)

        pred_tokens = self.model(audio, n_frames=target_params.shape[1])
        stylised = style + pred_tokens
        pred_params = self.fc(stylised)

        recon_loss = self.loss(pred_params, target_params)
        vel_loss = self.loss(pred_params[:, :-1] - pred_params[:, 1:], target_params[:, :-1] - target_params[:, 1:])

        loss = recon_loss + (10 * vel_loss)
        self.log('train_loss', loss)
        self.log('train_recon_loss', recon_loss)
        self.log('train_vel_loss', vel_loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

from Datasets import DubbingDataset, DataTypes
from pytorch_lightning.callbacks import ModelSummary

train_dataloader = torch.utils.data.DataLoader(
    DubbingDataset('C:/Users/jacks/Documents/Data/DubbingForExtras/v3/index.csv',
        data_types=[DataTypes.Audio, DataTypes.Params], T=11),
    batch_size=1,
    shuffle=True,
    num_workers=0,
)

val_dataloader = torch.utils.data.DataLoader(
    DubbingDataset('C:/Users/jacks/Documents/Data/DubbingForExtras/v3/index.csv',
        data_types=[DataTypes.Audio, DataTypes.Params], split='test', T=11),
    batch_size=1,
    shuffle=True,
    num_workers=0
)

model = Audio2Expression()
logger = WandbLogger(project='Audio2Expression')
logger = None
trainer = pl.Trainer(gpus=1, max_epochs=1, gradient_clip_val=0.5,
                     callbacks=[ModelSummary(max_depth=-1)], logger=logger)
trainer.fit(model, train_dataloader)
