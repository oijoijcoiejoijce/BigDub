import math
import torch
import torch.nn as nn


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz))) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(src, tgt):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len).to(src.device)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=src.device).type(torch.bool)

    return src_mask, tgt_mask


class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])


class Imitator(nn.Module):

    def __init__(self, n_params, n_hidden=64):
        super(Imitator, self).__init__()

        self.n_params = n_params
        self.n_hidden = n_hidden

        # Wav2Vec2 Part
        self.audio_feature_map = nn.Linear(768, n_hidden)

        # Transformer Part: Person Generic Visemes
        self.transformer = nn.Transformer(n_hidden, batch_first=False, dim_feedforward=128, nhead=4)
        self.start_token = nn.Parameter(torch.randn(1, 1, n_hidden), requires_grad=False)
        self.pe = PositionalEncoding(n_hidden, 0.1)

        self.fc = nn.Linear(n_hidden, n_hidden)

        # Person Specific Decoder
        self.decoder = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),

            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),

            nn.Linear(n_hidden, n_hidden),
            nn.ReLU()
        )

    def forward(self, audio, n_frames=None):
        audio_enc = self.audio_feature_map(audio)
        audio_enc = self.pe(audio_enc)
        # audio_enc = torch.cat([self.start_token.repeat(audio_enc.shape[0], 1, 1).to(audio.device), audio_enc], dim=1)

        if n_frames is None:
            n_frames = int(audio_enc.shape[0] * (30/50))

        target_in = self.start_token.repeat(1, audio_enc.shape[0], 1)
        audio_enc = audio_enc.permute(1, 0, 2)

        for i in range(n_frames):

            target_in_enc = self.pe(target_in)
            src_mask, tgt_mask = create_mask(audio_enc, target_in)
            target_pred = self.transformer(audio_enc, target_in_enc, tgt_mask=tgt_mask, src_mask=src_mask)
            target_pred = self.fc(target_pred)
            target_in = torch.cat([target_in, target_pred[-1:, :, :]], dim=0)

        audio_enc = audio_enc.permute(1, 0, 2)
        target_out = target_in[1:, :, :].permute(1, 0, 2)
        target_out = self.decoder(target_out)

        return target_out
