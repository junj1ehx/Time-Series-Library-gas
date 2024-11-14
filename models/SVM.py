
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding

class Model(nn.Module):
    """
    SVM model for time series forecasting
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.d_model = configs.d_model
        self.dropout = configs.dropout if hasattr(configs, 'dropout') else 0.1

        # Encoding
        self.enc_embedding = DataEmbedding(self.enc_in, self.d_model, configs.embed, configs.freq, configs.dropout)
        
        # SVM Layer
        self.svm_layer = nn.Linear(self.d_model, self.enc_in)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        
        # SVM layer
        dec_out = self.svm_layer(enc_out[:,-self.pred_len:,:])
        
        return dec_out  # [B, L, D]
