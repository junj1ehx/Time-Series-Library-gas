import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding

class RBFKernel(nn.Module):
    def __init__(self, gamma=1.0):
        super(RBFKernel, self).__init__()
        self.gamma = nn.Parameter(torch.tensor(gamma))

    def forward(self, x1, x2):
        distances = torch.sum((x1.unsqueeze(1) - x2.unsqueeze(0)) ** 2, dim=-1)
        return torch.exp(-self.gamma * distances)

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.d_model = configs.d_model
        
        # Hyperparameters
        self.C = 1.0
        self.epsilon = 0.1
        
        # Encoding
        self.enc_embedding = DataEmbedding(self.enc_in, self.d_model, 
                                         configs.embed, configs.freq, 
                                         configs.dropout)
        
        # Feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Linear(self.d_model, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64)
        )
        
        # Kernel
        self.kernel = RBFKernel(gamma=0.1)
        
        # Support vectors
        n_support_vectors = 50
        self.support_vectors = nn.Parameter(torch.randn(n_support_vectors, 64))
        self.alphas = nn.Parameter(torch.randn(n_support_vectors))
        
        # Output layer
        self.output_layer = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, self.enc_in)
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        
        # Get the sequence for prediction
        batch_size = enc_out.size(0)
        x = enc_out[:, -self.pred_len:, :]
        seq_len = x.size(1)
        
        # Reshape for feature extraction
        x = x.reshape(-1, self.d_model)
        
        # Extract features
        features = self.feature_extractor(x)
        
        # Compute kernel between input and support vectors
        kernel_matrix = self.kernel(features, self.support_vectors)
        
        # Compute SVM predictions
        svm_out = torch.matmul(kernel_matrix, self.alphas)
        svm_out = svm_out.unsqueeze(-1)  # Add channel dimension
        
        # Transform to final output
        predictions = self.output_layer(svm_out)
        
        # Reshape back to sequence format
        predictions = predictions.view(batch_size, seq_len, self.enc_in)
        
        return predictions

    def compute_loss(self, pred, true):
        """
        Compute SVR loss with epsilon-insensitive tube and regularization
        """
        # Epsilon-insensitive loss
        diff = pred - true
        epsilon_loss = torch.max(torch.zeros_like(diff), torch.abs(diff) - self.epsilon)
        
        # L2 regularization
        reg_loss = (torch.sum(self.alphas ** 2) + 
                   torch.sum(self.support_vectors ** 2) +
                   torch.sum(self.kernel.gamma ** 2))
        
        # Total loss
        total_loss = torch.mean(epsilon_loss) + self.C * reg_loss
        
        return total_loss

def adjust_learning_rate(optimizer, epoch, args):
    # Learning rate decay
    lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch-1) // 1))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))

class SVMTrainer:
    def __init__(self, model, learning_rate=0.001):
        self.model = model
        self.optimizer = torch.optim.Adam([
            {'params': model.parameters(), 'lr': learning_rate},
            {'params': [model.output_scale], 'lr': learning_rate * 0.1},
            {'params': [model.output_bias], 'lr': learning_rate * 0.1}
        ])

    def train_step(self, batch_x, batch_y):
        self.model.train()
        self.optimizer.zero_grad()
        
        predictions = self.model(batch_x)
        loss = self.model.compute_loss(predictions, batch_y)
        
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # Project coefficients to be non-negative
        with torch.no_grad():
            self.model.alphas_pos.data.clamp_(min=0)
            self.model.alphas_neg.data.clamp_(min=0)
        
        return loss.item()