import torch
import torch.nn as nn

class TransformerModel(nn.Module):
    def __init__(self, input_dim, model_dim, n_heads, n_layers, output_dim):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, model_dim)
        self.transformer = nn.Transformer(d_model=model_dim, nhead=n_heads, num_encoder_layers=n_layers)
        self.fc = nn.Linear(model_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = self.fc(x)
        return x