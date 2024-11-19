import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class Generator(nn.Module):
    def __init__(self, input_channels, output_channel, d_model=300, nhead=10, num_layers=8, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        
        # Initial projection from input channels to d_model
        self.input_proj = nn.Linear(input_channels, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer encoder with layer normalization
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, 
                                                   nhead=nhead,
                                                   dim_feedforward=dim_feedforward,
                                                   dropout=dropout,
                                                   batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_proj = nn.Linear(d_model, output_channel)  # Same channels as input
        
        # Layer Normalization
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Dropout layers
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x shape: [batch_size, channels, seq_len]
        x = x.transpose(1, 2)  # [batch_size, seq_len, channels]
        
        # Project to transformer dimensions
        x = self.input_proj(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Pass through transformer
        x = self.transformer_encoder(x)
        
        # Apply layer normalization and dropout
        x = self.layer_norm(x)
        x = self.dropout(x)
        
        # Project back to original channel dimension
        x = self.output_proj(x)
        
        # Restore original shape
        x = x.transpose(1, 2)  # [batch_size, channels, seq_len]
        return x


class Discriminator(nn.Module):
    def __init__(self, input_channels, d_model=256, nhead=8, num_layers=4, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        
        # Initial projection
        self.input_proj = nn.Linear(input_channels * 2, d_model)  # *2 because we concat real/fake pairs
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer encoder with layer normalization
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model,
                                                   nhead=nhead,
                                                   dim_feedforward=dim_feedforward,
                                                   dropout=dropout,
                                                   batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output layers
        self.fc1 = nn.Linear(d_model, d_model // 2)
        self.fc2 = nn.Linear(d_model // 2, 1)
        self.relu = nn.LeakyReLU(0.2)  # Leaky ReLU activation
        
        # Layer Normalization
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Dropout layers
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, y):
        # x, y shape: [batch_size, channels, seq_len]
        x = x.transpose(1, 2)  # [batch_size, seq_len, channels]
        y = y.transpose(1, 2)
        
        # Concatenate input and output signals
        xy = torch.cat([x, y], dim=-1)  # [batch_size, seq_len, channels*2]
        
        # Project to transformer dimensions
        xy = self.input_proj(xy)
        
        # Add positional encoding
        xy = self.pos_encoder(xy)
        
        # Pass through transformer
        xy = self.transformer_encoder(xy)
        
        # Apply layer normalization and dropout
        xy = self.layer_norm(xy)
        xy = self.dropout(xy)
        
        # Global average pooling
        xy = torch.mean(xy, dim=1)  # [batch_size, d_model]
        
        # Final classification
        xy = self.relu(self.fc1(xy))
        xy = self.fc2(xy)
        return xy