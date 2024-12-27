import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding

class Model(nn.Module):
    """
    ANN model for time series forecasting
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.d_model = configs.d_model
        self.dropout = configs.dropout if hasattr(configs, 'dropout') else 0.1
        self.num_layers = configs.e_layers if hasattr(configs, 'e_layers') else 2

        # Encoding
        self.enc_embedding = DataEmbedding(self.enc_in, self.d_model, configs.embed, configs.freq, configs.dropout)
        
        # ANN Layers
        self.hidden_layers = nn.ModuleList()
        for _ in range(self.num_layers):
            self.hidden_layers.append(nn.Linear(self.d_model, self.d_model))
            self.hidden_layers.append(nn.ReLU())
            # self.hidden_layers.append(nn.Dropout(self.dropout))

        # Output projection
        self.projection = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, self.enc_in * self.pred_len)
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        
        # ANN layers
        for layer in self.hidden_layers:
            enc_out = layer(enc_out)
        
        # Reshape & Prediction
        dec_out = self.projection(enc_out[:,-1:,:])
        dec_out = dec_out.view(dec_out.size(0), self.pred_len, self.enc_in)
        
        return dec_out  # [B, L, D]



