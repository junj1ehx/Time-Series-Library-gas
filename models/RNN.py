import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding

class Model(nn.Module):
    """
    RNN model for time series forecasting
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.d_model = configs.d_model
        self.dropout = configs.dropout if hasattr(configs, 'dropout') else 0.1
        self.num_layers = configs.e_layers if hasattr(configs, 'e_layers') else 1

        # Encoding
        self.enc_embedding = DataEmbedding(self.enc_in, self.d_model, configs.embed, configs.freq, configs.dropout)
        
        # RNN Encoder
        self.encoder = nn.RNN(
            input_size=self.d_model,
            hidden_size=self.d_model,
            num_layers=self.num_layers,
            batch_first=True,
            # dropout=self.dropout
        )

        # Output projection
        self.projection = nn.Linear(self.d_model, self.enc_in)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        
        # RNN encoder
        enc_out, _ = self.encoder(enc_out)
        
        # Reshape & Prediction
        dec_out = self.projection(enc_out[:,-self.pred_len:,:])
        
        return dec_out  # [B, L, D]
