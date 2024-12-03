
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding

class RBFKernel(nn.Module):
    def __init__(self, gamma=1.0):
        super(RBFKernel, self).__init__()
        self.gamma = gamma

    def forward(self, x1, x2):
        # Compute RBF (Gaussian) kernel: K(x1, x2) = exp(-gamma * ||x1 - x2||^2)
        distances = torch.sum((x1.unsqueeze(1) - x2.unsqueeze(0)) ** 2, dim=-1)
        return torch.exp(-self.gamma * distances)

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

        # Hyperparameters
        self.C = 1.0  # Regularization parameter
        self.epsilon = 0.1  # Epsilon for SVR
        
        self.kernel = RBFKernel(gamma=0.1)
        
        # Support vectors will be learned during training
        self.support_vectors = nn.Parameter(torch.randn(50, self.d_model))
        self.alphas = nn.Parameter(torch.zeros(50))
        self.bias = nn.Parameter(torch.zeros(self.enc_in))


        # Encoding
        self.enc_embedding = DataEmbedding(self.enc_in, self.d_model, configs.embed, configs.freq, configs.dropout)
        
        # SVM Layer
        self.svm_layer = nn.Linear(self.d_model, self.enc_in)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        
        # SVM layer
        # dec_out = self.svm_layer(enc_out[:,-self.pred_len:,:])

         # Get the last sequence for prediction
        x = enc_out[:, -self.pred_len:, :]
        batch_size, seq_len, _ = x.shape
        x = x.reshape(-1, self.d_model)
        
        # Compute kernel between input and support vectors
        kernel_matrix = self.kernel(x, self.support_vectors)

        # SVM prediction
        predictions = torch.mm(kernel_matrix, self.alphas.unsqueeze(1)) + self.bias
        
        # Reshape back to original dimensions
        predictions = predictions.view(batch_size, seq_len, self.enc_in)
        
        return predictions


        return dec_out  # [B, L, D]
